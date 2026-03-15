import os
import sys
import math
import random
from collections import defaultdict

import cv2
import numpy as np
import shapely
import shapely.geometry
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from tqdm import tqdm
from shapely.geometry import Polygon, MultiPoint

from models.dark_net import Darknet
from models.ET_haa import ET
from models.vln_model import CustomBERTModel
from transformers import BertTokenizerFast
from utils.logger import print_progress


def debug_memory():
    import collections
    import gc
    import resource

    print('maxrss = {}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    tensors = collections.Counter(
        (str(o.device), o.dtype, tuple(o.shape))
        for o in gc.get_objects()
        if torch.is_tensor(o)
    )
    for line in tensors.items():
        print('{}\t{}'.format(*line))


def compute_iou(a, b):
    a = np.array(a)
    b = np.array(b)
    poly1 = Polygon(a).convex_hull
    poly2 = Polygon(b).convex_hull
    union_poly = np.concatenate((a, b))
    if not poly1.intersects(poly2):
        return 0
    try:
        inter_area = poly1.intersection(poly2).area
        union_area = MultiPoint(union_poly).convex_hull.area
        if union_area == 0:
            return 0
        return float(inter_area) / union_area
    except shapely.geos.TopologicalError:
        print('shapely.geos.TopologicalError occured, iou set to 0')
        return 0


def is_default_gpu(opts) -> bool:
    return opts.local_rank == -1 or dist.get_rank() == 0


def get_direction(start, end):
    vec = np.array(end) - np.array(start)
    if vec[1] > 0:
        angle = np.arctan(vec[0] / vec[1]) / 1.57 * 90
    elif vec[1] < 0:
        angle = np.arctan(vec[0] / vec[1]) / 1.57 * 90 + 180
    else:
        angle = 90 if np.sign(vec[0]) == 1 else 270
    return (360 - angle + 90) % 360


def extract_intersection_coords(geometry):
    if geometry is None or geometry.is_empty:
        return []
    try:
        return list(geometry.coords)
    except (NotImplementedError, AttributeError):
        pass
    coords = []
    if hasattr(geometry, 'geoms'):
        for geom in geometry.geoms:
            coords.extend(extract_intersection_coords(geom))
    return coords


class NavCMTAgent:
    def __init__(self, args, rank=0):
        self.results = {}
        self.losses = []
        self.args = args
        self.env = []
        self.env_name = ''
        self.feedback = 'student'
        self.rank = rank
        random.seed(1)
        if self.args.local_rank != -1:
            device_index = self.args.local_rank
        elif torch.cuda.is_available():
            device_index = torch.cuda.current_device()
        else:
            device_index = 0
        self.device = torch.device('cuda', device_index)

        self.rgb_mean = np.array([60.134, 49.697, 40.746], dtype=np.float32).reshape((3, 1, 1))
        self.rgb_std = np.array([29.99, 24.498, 22.046], dtype=np.float32).reshape((3, 1, 1))
        self.default_gpu = is_default_gpu(self.args)

        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.lang_model = CustomBERTModel().to(self.device)
        self.img_tensor = transforms.ToTensor()

        self.vision_model = Darknet(self.args.darknet_model_file, 224).to(self.device)
        new_state = torch.load(self.args.darknet_weight_file, map_location='cpu')
        state = self.vision_model.state_dict()
        model_keys = set(state.keys())
        state_dict = {k: v for k, v in new_state['model'].items() if k in model_keys}
        state.update(state_dict)
        self.vision_model.load_state_dict(state)

        self.vln_model = ET(self.args).to(self.device)
        self._maybe_wrap_ddp()

        optimizer_map = {
            'adam': torch.optim.Adam,
            'adamW': torch.optim.AdamW,
            'rms': torch.optim.RMSprop,
            'sgd': torch.optim.SGD,
        }
        optimizer_cls = optimizer_map[args.optim]
        self.et_optimizer = optimizer_cls(filter(lambda p: p.requires_grad, self.vln_model.parameters()), lr=args.lr)
        self.lang_model_optimizer = optimizer_cls(filter(lambda p: p.requires_grad, self.lang_model.parameters()), lr=args.lr)
        self.vision_model_optimizer = optimizer_cls(filter(lambda p: p.requires_grad, self.vision_model.parameters()), lr=args.lr)
        self.rl_optimizer = optimizer_cls(filter(lambda p: p.requires_grad, self.vln_model.parameters()), lr=args.rl_lr)

        self.progress_regression = nn.MSELoss(reduction='sum')
        self.stop_criterion = nn.BCEWithLogitsLoss(reduction='sum')
        self.loss = 0
        self.logs = defaultdict(list)
        sys.stdout.flush()

    def _maybe_wrap_ddp(self):
        if self.args.world_size <= 1:
            return
        ddp_kwargs = {
            'device_ids': [self.device.index],
            'output_device': self.device.index,
            'find_unused_parameters': False,
            'broadcast_buffers': False,
        }
        self.lang_model = DDP(self.lang_model, **ddp_kwargs)
        self.vision_model = DDP(self.vision_model, **ddp_kwargs)
        self.vln_model = DDP(self.vln_model, **ddp_kwargs)

    def _unwrap(self, model):
        return model.module if isinstance(model, DDP) else model

    def get_results(self):
        return self.results

    def test(self, loader, env_name='no_name_provided', feedback='student', not_in_train=False, **kwargs):
        self.feedback = feedback
        self.env_name = env_name

        self.vln_model.eval()
        self.lang_model.eval()
        self.vision_model.eval()

        self.losses = []
        self.results = {}
        self.loss = 0
        for _ in loader:
            for traj in self.rollout(not_in_train=True, **kwargs):
                self.loss = 0
                self.results[traj['instr_id']] = traj

    def train(self, loader, n_epochs, feedback='student', nss_w_weighting=1, **kwargs):
        self.feedback = feedback
        self.lang_model.train()
        self.vln_model.train()
        self.vision_model.train()
        self.losses = []

        for epoch in range(1, n_epochs + 1):
            for _ in tqdm(loader, disable=not self.default_gpu):
                self.lang_model_optimizer.zero_grad()
                self.vision_model_optimizer.zero_grad()
                self.et_optimizer.zero_grad()
                self.loss = 0

                if feedback == 'teacher':
                    self.feedback = 'teacher'
                    self.rollout(train_ml=self.args.teacher_weight, train_rl=False, nss_w=self.args.nss_w * nss_w_weighting, **kwargs)
                elif feedback == 'student':
                    self.feedback = 'teacher'
                    self.rollout(train_ml=self.args.ml_weight, train_rl=False, nss_w=0, **kwargs)
                    self.feedback = 'student'
                    self.rollout(train_ml=self.args.ml_weight, train_rl=False, nss_w=self.args.nss_w * nss_w_weighting, **kwargs)
                else:
                    raise ValueError('Unsupported feedback mode: %s' % feedback)

                self.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vln_model.parameters(), 40.)
                self.lang_model_optimizer.step()
                self.vision_model_optimizer.step()
                self.et_optimizer.step()

            if self.default_gpu:
                print_progress(epoch, n_epochs, prefix='Progress:', suffix='Complete', bar_length=50)

    def train_ppo(self, loader, n_iters):
        if n_iters <= 0:
            return

        self.feedback = 'student'
        self.lang_model.eval()
        self.vision_model.eval()
        self.vln_model.eval()

        updates = 0
        while updates < n_iters:
            collected = []
            collected_episodes = 0
            for _ in tqdm(loader, disable=not self.default_gpu):
                transitions, rollout_stats = self.rollout_ppo()
                collected.extend(transitions)
                collected_episodes += rollout_stats.get('episodes', 0)
                self.logs['ppo_rollout_return'].append(rollout_stats.get('mean_return', 0.0))
                self.logs['ppo_rollout_len'].append(rollout_stats.get('mean_len', 0.0))

                if collected_episodes < self.args.ppo_batch_episodes:
                    continue

                self._run_ppo_update(collected)
                collected = []
                collected_episodes = 0
                updates += 1
                if updates >= n_iters:
                    break

            if collected and updates < n_iters:
                self._run_ppo_update(collected)
                updates += 1

    def _run_ppo_update(self, transitions):
        ppo_stats = self.update_ppo(transitions)
        for key, value in ppo_stats.items():
            self.logs[key].append(value)

        anchor_stats = self.run_ppo_il_anchor()
        for key, value in anchor_stats.items():
            self.logs[key].append(value)

    def NSS(self, sal, fix):
        m = torch.mean(sal.view(-1, 224 * 224), 1).view(-1, 1, 1)
        std = torch.std(sal.view(-1, 224 * 224), 1).view(-1, 1, 1)
        if self.args.nss_r == 0:
            n_sal = (sal - m) / std
        elif self.args.nss_r == 1:
            n_sal = (sal - m) / std / 2 + 1
        else:
            n_sal = (sal - m) / std / 2 - 1

        s_fix = torch.sum(fix.view(-1, 224 * 224), 1) + 0.001
        ns = n_sal * fix
        s_ns = torch.sum(ns.view(-1, 224 * 224), 1)
        return -torch.mean(s_ns / s_fix)

    def move_view_corners(self, corners, angle, distance, altitude, gps_botm_left, gps_top_right, input_current_direction=None):
        def move_view_corner_forward(cs, change):
            new_cs = np.zeros((4, 2))
            new_cs[0] = cs[0] + (cs[0] - cs[3]) / np.linalg.norm((cs[3] - cs[0])) * change
            new_cs[1] = cs[1] + (cs[1] - cs[2]) / np.linalg.norm((cs[2] - cs[1])) * change
            new_cs[2] = cs[2] + (cs[1] - cs[2]) / np.linalg.norm((cs[2] - cs[1])) * change
            new_cs[3] = cs[3] + (cs[0] - cs[3]) / np.linalg.norm((cs[3] - cs[0])) * change
            return new_cs

        def rotation_anticlock(theta, p):
            m = np.array([
                [np.cos(theta / 180 * 3.14159), np.sin(theta / 180 * 3.14159)],
                [-np.sin(theta / 180 * 3.14159), np.cos(theta / 180 * 3.14159)],
            ])
            return np.matmul(m, np.array([p[0], p[1]]))

        def change_corner(cs, change):
            new_cs = np.zeros((4, 2))
            new_cs[0] = cs[0] + (cs[0] - cs[1]) / np.linalg.norm((cs[1] - cs[0])) * change
            new_cs[0] += (cs[0] - cs[3]) / np.linalg.norm((cs[3] - cs[0])) * change
            new_cs[1] = cs[1] + (cs[1] - cs[0]) / np.linalg.norm((cs[1] - cs[0])) * change
            new_cs[1] += (cs[1] - cs[2]) / np.linalg.norm((cs[2] - cs[1])) * change
            new_cs[2] = cs[2] + (cs[2] - cs[3]) / np.linalg.norm((cs[2] - cs[3])) * change
            new_cs[2] += (cs[2] - cs[1]) / np.linalg.norm((cs[2] - cs[1])) * change
            new_cs[3] = cs[3] + (cs[3] - cs[2]) / np.linalg.norm((cs[2] - cs[3])) * change
            new_cs[3] += (cs[3] - cs[0]) / np.linalg.norm((cs[3] - cs[0])) * change
            return new_cs

        current_direction = round(get_direction(np.mean(corners, axis=0), (corners[0] + corners[1]) / 2)) % 360
        if input_current_direction is not None and abs(input_current_direction - current_direction) > 2:
            print('warning, currencting the view area by: +', input_current_direction - current_direction)
            angle += input_current_direction

        current_view_area_edge_length = np.linalg.norm((corners[1]) - corners[0]) * 11.13 * 1e4
        step_change_of_view_zoom = 0.5 * (altitude - current_view_area_edge_length) / 11.13 / 1e4
        new_corners = []
        for point in change_corner(corners, step_change_of_view_zoom):
            if point[0] > gps_botm_left[0] and point[0] < gps_top_right[0] and point[1] > gps_botm_left[1] and point[1] < gps_top_right[1]:
                new_corners.append(point)
            else:
                break
        if len(new_corners) != 4:
            return np.array(corners), current_direction
        corners = new_corners

        mean_im_coords = np.mean(corners, axis=0)
        centered = [corners[0] - mean_im_coords, corners[1] - mean_im_coords, corners[2] - mean_im_coords, corners[3] - mean_im_coords]
        rotated_corners = []
        for point in centered:
            rotated_point = mean_im_coords + rotation_anticlock(-angle, point)
            if rotated_point[0] > gps_botm_left[0] and rotated_point[0] < gps_top_right[0] and rotated_point[1] > gps_botm_left[1] and rotated_point[1] < gps_top_right[1]:
                rotated_corners.append(rotated_point)
            else:
                break
        if len(rotated_corners) != 4:
            return np.array(corners), current_direction

        new_corners = []
        for point in move_view_corner_forward(np.array(rotated_corners), distance):
            if point[0] > gps_botm_left[0] and point[0] < gps_top_right[0] and point[1] > gps_botm_left[1] and point[1] < gps_top_right[1]:
                new_corners.append(point)
            else:
                break
        if len(new_corners) != 4:
            return np.array(rotated_corners), (current_direction + angle) % 360
        return np.array(new_corners), (current_direction + angle) % 360

    def teacher_action(self, obs, ended, corners, directions):
        teacher_a = [['0', '0'] for _ in range(len(obs))]
        progress = np.zeros((len(obs), 1), dtype=np.float32)
        for i in range(len(obs)):
            current_pos = np.mean(corners[i], axis=0)
            iou = compute_iou(corners[i], obs[i]['gt_path_corners'][-1])
            progress[i] = np.float32(iou)

            min_dis = 1000
            closest_step_index = len(obs[i]['gt_path_corners']) - 1
            for j in range(len(obs[i]['gt_path_corners']) - 1, -1, -1):
                gt_pos = np.mean(obs[i]['gt_path_corners'][j], axis=0)
                dis_to_current = np.linalg.norm(gt_pos - current_pos)
                if dis_to_current + 0.00001 < min_dis:
                    min_dis = dis_to_current
                    closest_step_index = j

            teacher_a[i][1] = float(
                (np.linalg.norm(obs[i]['gt_path_corners'][closest_step_index][0] - obs[i]['gt_path_corners'][closest_step_index][1]) * 11.13 * 1e4 - 40) / (400 - 40)
            )
            if ended[i] or progress[i] > self.args.stop_iou_th:
                teacher_a[i][0] = np.array([0, 0], dtype=np.float32)
                continue

            goal_corner_center = np.mean(obs[i]['gt_path_corners'][-1], axis=0)
            teacher_a[i][0] = goal_corner_center
            shapely_poly = shapely.geometry.Polygon(corners[i])
            if self.feedback == 'student':
                line = [current_pos, np.mean(obs[i]['gt_path_corners'][-1], axis=0)]
                intersection_line = extract_intersection_coords(
                    shapely_poly.intersection(shapely.geometry.LineString(line))
                )
            else:
                line = [np.mean(obs[i]['gt_path_corners'][j], axis=0) for j in range(len(obs[i]['gt_path_corners']))]
                intersection = shapely_poly.intersection(shapely.geometry.LineString(line))
                intersection_line = extract_intersection_coords(intersection)
                if intersection_line == []:
                    line = [current_pos, np.mean(obs[i]['gt_path_corners'][-1], axis=0)]
                    intersection_line = extract_intersection_coords(
                        shapely_poly.intersection(shapely.geometry.LineString(line))
                    )

            min_distance = 1
            for x in intersection_line:
                x = np.array(x)
                distance = np.linalg.norm(x - goal_corner_center)
                if distance < min_distance:
                    min_distance = distance
                    teacher_a[i][0] = x

            net_next_pos = 1e5 * (teacher_a[i][0] - current_pos)
            net_y = np.round(1e5 * ((corners[i][0] + corners[i][1]) / 2 - current_pos)).astype(np.int64)
            net_x = np.round(1e5 * ((corners[i][1] + corners[i][2]) / 2 - current_pos)).astype(np.int64)
            a = np.mat([[net_x[0], net_y[0]], [net_x[1], net_y[1]]])
            b = np.mat([net_next_pos[0], net_next_pos[1]]).T
            r = np.linalg.solve(a, b)
            gt_next_pos_ratio = [r[0, 0], r[1, 0]]

            max_abs = max(abs(gt_next_pos_ratio[0]), abs(gt_next_pos_ratio[1]), 1)
            gt_next_pos_ratio[0] /= max_abs
            gt_next_pos_ratio[1] /= max_abs
            teacher_a[i][0] = np.array(gt_next_pos_ratio, dtype=np.float32)
        return teacher_a, progress

    def gps_to_img_coords(self, gps, gps_botm_left, gps_top_right, lat_ratio):
        return int(round((gps[1] - gps_botm_left[1]) / lat_ratio)), int(round((gps_top_right[0] - gps[0]) / lat_ratio))

    def prepare_language_features(self, obs):
        lang_inputs = ['' if self.args.vision_only else ob['instructions'] for ob in obs]
        encoding = self.tokenizer(lang_inputs, padding=True, return_tensors='pt')
        lang_features, linear_cls, _ = self.lang_model(
            encoding['input_ids'].to(self.device),
            encoding['attention_mask'].to(self.device)
        )

        dialog_inputs = list(lang_inputs)
        if not self.args.train_val_on_full:
            dialog_inputs = [ob['pre_dialogs'] + ob['instructions'] for ob in obs]
            encoding = self.tokenizer(dialog_inputs, padding=True, return_tensors='pt')
            _, linear_cls, _ = self.lang_model(
                encoding['input_ids'].to(self.device),
                encoding['attention_mask'].to(self.device)
            )
        return lang_features, linear_cls, dialog_inputs

    def init_rollout_inputs(self, batch_size, lang_features, linear_cls):
        return {
            'directions': torch.zeros((batch_size, 0, 2), device=self.device),
            'frames': torch.zeros(batch_size, 0, 512, 49, device=self.device),
            'lenths': [0 for _ in range(batch_size)],
            'lang': lang_features,
            'lang_cls': linear_cls,
        }

    def initialize_traj(self, obs, dialog_inputs):
        traj = [defaultdict(list) for _ in obs]
        for i, ob in enumerate(obs):
            traj[i]['instr_id'] = ob['map_name'] + '__' + ob['route_index']
            rounds = dialog_inputs[i].split('[QUE]')
            remove = sum(1 for r in rounds if 'Yes' in r[0:5])
            traj[i]['num_dia'] = len(rounds) - remove
            traj[i]['path_corners'] = [(np.array(ob['gt_path_corners'][0]), ob['starting_angle'])]
            traj[i]['gt_path_corners'] = ob['gt_path_corners']
        return traj

    def extract_visual_features(self, obs):
        images = np.stack([ob['current_view'].copy() for ob in obs])[:, :, :, ::-1].transpose(0, 3, 1, 2)
        images = np.ascontiguousarray(images, dtype=np.float32)
        images -= self.rgb_mean
        images /= self.rgb_std
        im_feature = self.vision_model(torch.from_numpy(images).to(self.device))
        return im_feature.view(im_feature.size(0), im_feature.size(1), -1)

    def append_step_inputs(self, rollout_inputs, im_feature, direction_t, ended):
        current_direct = direction_t.view(-1, 1).to(self.device)
        direction = torch.concat((torch.sin(current_direct / 180 * 3.14159), torch.cos(current_direct / 180 * 3.14159)), axis=1)

        if self.args.no_direction:
            rollout_inputs['directions'] = torch.hstack((rollout_inputs['directions'], torch.zeros_like(direction.view(-1, 1, 2))))
        else:
            rollout_inputs['directions'] = torch.hstack((rollout_inputs['directions'], direction.view(-1, 1, 2)))

        if self.args.language_only:
            rollout_inputs['frames'] = torch.hstack((rollout_inputs['frames'], torch.zeros_like(im_feature.view(-1, 1, 512, 49))))
        else:
            rollout_inputs['frames'] = torch.hstack((rollout_inputs['frames'], im_feature.view(-1, 1, 512, 49)))

        for i in range(len(ended)):
            if not ended[i]:
                rollout_inputs['lenths'][i] += 1
        return rollout_inputs

    def forward_navigation(self, rollout_inputs):
        model_out, pred_saliency = self.vln_model(
            directions=rollout_inputs['directions'],
            frames=rollout_inputs['frames'],
            lenths=rollout_inputs['lenths'],
            lang=rollout_inputs['lang'],
            lang_cls=rollout_inputs['lang_cls'],
        )
        action = model_out['action']
        return {
            'action': action,
            'pred_next_pos_ratio': action[:, :2],
            'pred_altitude': action[:, 2],
            'pred_progress': model_out['progress'],
            'stop_logit': model_out['stop_logit'],
            'stop_prob': torch.sigmoid(model_out['stop_logit']),
            'state_value': model_out['state_value'],
            'action_log_std': model_out['action_log_std'],
        }, pred_saliency

    def normalize_waypoint(self, waypoint):
        waypoint = np.array(waypoint, dtype=np.float32, copy=True)
        for i in range(len(waypoint)):
            max_abs = max(abs(waypoint[i][0]), abs(waypoint[i][1]), 1.0)
            waypoint[i][0] /= max_abs
            waypoint[i][1] /= max_abs
        return waypoint

    def clip_altitude(self, altitude):
        altitude = np.array(altitude, dtype=np.float32, copy=True)
        for i in range(len(altitude)):
            altitude[i] = min(1., max(0., altitude[i]))
        return altitude

    def build_stop_label(self, gt_progress):
        gt_progress = np.asarray(gt_progress, dtype=np.float32).reshape(-1)
        return torch.from_numpy((gt_progress >= self.args.stop_iou_th).astype(np.float32)).to(self.device)

    def compute_il_losses(self, pred_dict, target, gt_progress, ended):
        device = pred_dict['pred_next_pos_ratio'].device
        valid_mask = (~torch.from_numpy(ended)).to(device=device, dtype=torch.float32)
        denom = valid_mask.sum().clamp(min=1.0)

        gt_next_pos_ratio = torch.from_numpy(np.stack([x[0] for x in target]).astype(np.float32)).to(device)
        gt_altitude = torch.tensor([x[1] for x in target], dtype=torch.float32, device=device)
        gt_progress_t = torch.from_numpy(np.asarray(gt_progress, dtype=np.float32).reshape(-1)).to(device)
        stop_label = self.build_stop_label(gt_progress).to(device)

        pred_next_pos_ratio = pred_dict['pred_next_pos_ratio']
        pred_altitude = pred_dict['pred_altitude']
        pred_progress = pred_dict['pred_progress']
        stop_logit = pred_dict['stop_logit']

        pred_angle = ((torch.atan2(pred_next_pos_ratio[:, 0], pred_next_pos_ratio[:, 1] + 1e-5) / math.pi) + 2) / 2 % 1
        gt_angle = ((torch.atan2(gt_next_pos_ratio[:, 0], gt_next_pos_ratio[:, 1] + 1e-5) / math.pi) + 2) / 2 % 1

        position_loss = (((pred_next_pos_ratio - gt_next_pos_ratio) ** 2).sum(dim=1) * valid_mask).sum()
        angle_loss = (((pred_angle - gt_angle) ** 2) * valid_mask).sum()
        altitude_loss = (((pred_altitude - gt_altitude) ** 2) * valid_mask).sum()
        progress_loss = (((pred_progress - gt_progress_t) ** 2) * valid_mask).sum()
        stop_loss = (F.binary_cross_entropy_with_logits(stop_logit, stop_label, reduction='none') * valid_mask).sum()

        total_loss = position_loss + angle_loss + altitude_loss + progress_loss + self.args.stop_loss_w * stop_loss
        stop_acc = ((((torch.sigmoid(stop_logit) >= self.args.stop_th).float() == stop_label).float()) * valid_mask).sum() / denom

        return total_loss, {
            'position_loss': float((position_loss / denom).detach().item()),
            'angle_loss': float((angle_loss / denom).detach().item()),
            'altitude_loss': float((altitude_loss / denom).detach().item()),
            'progress_loss': float((progress_loss / denom).detach().item()),
            'stop_loss': float((stop_loss / denom).detach().item()),
            'stop_acc': float(stop_acc.detach().item()),
        }, stop_label

    def compute_saliency_loss(self, pred_saliency, obs, traj, not_in_train, nss_w, t):
        saliency_loss = torch.tensor(0.0, device=pred_saliency.device)
        logged_nss = []
        for i in range(len(obs)):
            pred_saliency_cpu = pred_saliency[i].clip(0, 1).detach().cpu().numpy().reshape(224, 224, 1)
            gt_saliency = obs[i]['gt_saliency'].reshape(224, 224, 1)
            if np.sum(obs[i]['gt_saliency']) > 0:
                nss_loss = self.NSS(pred_saliency[i], torch.from_numpy(obs[i]['gt_saliency']).to(self.device))
                if not torch.isnan(nss_loss):
                    saliency_loss = saliency_loss + nss_w * nss_loss
                    logged_nss.append(nss_loss.detach().item())
                if not_in_train and self.feedback == 'teacher':
                    tp = np.sum(pred_saliency_cpu * gt_saliency, dtype=np.float32)
                    pred_sum = np.sum(pred_saliency_cpu, dtype=np.float32)
                    precision = tp / pred_sum if pred_sum != 0 else 0.
                    recall = tp / np.sum(gt_saliency, dtype=np.float32)
                    traj[i]['human_att_performance'].append([precision, recall])
                    if not torch.isnan(nss_loss):
                        traj[i]['nss'].append(nss_loss.item())

            if self.args.inference and self.feedback == 'teacher':
                pred_max = max(float(np.max(pred_saliency_cpu)), 1e-6)
                cv2.imwrite(
                    self.args.pred_dir + '/debug_images/' + self.env_name + 'val' + obs[i]['map_name'] + '_' + obs[i]['route_index'] + '_pred_att_' + str(t) + '.jpg',
                    cv2.applyColorMap(np.uint8(255 * (pred_saliency_cpu / pred_max)), cv2.COLORMAP_JET),
                )
                cv2.imwrite(
                    self.args.pred_dir + '/debug_images/' + self.env_name + 'val' + obs[i]['map_name'] + '_' + obs[i]['route_index'] + '_gt_att_' + str(t) + '.jpg',
                    cv2.applyColorMap(np.uint8(255 * gt_saliency), cv2.COLORMAP_JET),
                )
                cv2.imwrite(
                    self.args.pred_dir + '/debug_images/' + self.env_name + 'val' + obs[i]['map_name'] + '_' + obs[i]['route_index'] + '_input_' + str(t) + '.jpg',
                    obs[i]['current_view'],
                )
        return saliency_loss, float(np.mean(logged_nss)) if logged_nss else 0.0

    def should_stop(self, stop_prob, stop_label=None):
        if self.feedback == 'teacher' and stop_label is not None:
            return stop_label.detach().cpu().numpy() >= 0.5
        return stop_prob.detach().cpu().numpy() >= self.args.stop_th

    def execute_action(self, corners, current_direction, action, obs_item):
        a_direction = (math.atan2(action[0][0], action[0][1]) / 3.14159 + 2) / 2 % 1
        a_distance = np.linalg.norm(action[0]) * (np.linalg.norm(corners[0] - corners[1]) / 2)
        a_altitude = float(min(1., max(0., action[1])))
        return self.move_view_corners(
            corners,
            round(a_direction * 360),
            a_distance,
            round(a_altitude * 360) + 40,
            obs_item['gps_botm_left'],
            obs_item['gps_top_right'],
            current_direction,
        )

    def compute_reward(self, prev_iou, current_iou, stop_action, timeout, success_stop):
        reward = self.args.step_penalty
        reward += self.args.delta_iou_reward * (current_iou - prev_iou)
        if stop_action:
            reward += self.args.success_stop_reward if success_stop else self.args.false_stop_penalty
        elif prev_iou >= self.args.stop_iou_th:
            reward += self.args.overshoot_penalty
        if timeout and not success_stop:
            reward += self.args.max_step_penalty
        return reward

    def snapshot_transition(self, rollout_inputs, idx):
        length = max(int(rollout_inputs['lenths'][idx]), 1)
        return {
            'directions': rollout_inputs['directions'][idx:idx + 1, :length].detach().cpu(),
            'frames': rollout_inputs['frames'][idx:idx + 1, :length].detach().cpu(),
            'lenths': [length],
            'lang': rollout_inputs['lang'][idx:idx + 1].detach().cpu(),
            'lang_cls': rollout_inputs['lang_cls'][idx:idx + 1].detach().cpu(),
        }

    def policy_action_to_env_action(self, action_tensor):
        action_np = action_tensor.detach().cpu().numpy()
        return self.normalize_waypoint(action_np[:, :2]), self.clip_altitude(action_np[:, 2])

    def recompute_policy_metrics(self, transition):
        model_out, _ = self.vln_model(
            directions=transition['directions'].to(self.device),
            frames=transition['frames'].to(self.device),
            lenths=transition['lenths'],
            lang=transition['lang'].to(self.device),
            lang_cls=transition['lang_cls'].to(self.device),
        )
        action = model_out['action']
        action_mean = torch.cat((action[:, :2], action[:, 2:3]), dim=1)
        action_std = model_out['action_log_std'].exp().unsqueeze(0).expand_as(action_mean)
        action_dist = Normal(action_mean, action_std)
        stop_dist = Bernoulli(logits=model_out['stop_logit'])

        sampled_action = transition['sampled_action'].to(self.device).unsqueeze(0)
        sampled_stop = torch.tensor([transition['sampled_stop']], dtype=torch.float32, device=self.device)
        log_prob = action_dist.log_prob(sampled_action).sum(dim=1) + stop_dist.log_prob(sampled_stop)
        entropy = action_dist.entropy().sum(dim=1) + stop_dist.entropy()
        return log_prob.squeeze(0), model_out['state_value'].squeeze(0), entropy.squeeze(0)

    def compute_episode_gae(self, episode):
        gae = 0.0
        next_value = 0.0
        for step in reversed(range(len(episode))):
            reward = episode[step]['reward']
            done = episode[step]['done']
            value = episode[step]['value']
            non_terminal = 1.0 - float(done)
            delta = reward + self.args.ppo_gamma * next_value * non_terminal - value
            gae = delta + self.args.ppo_gamma * self.args.ppo_lam * non_terminal * gae
            episode[step]['advantage'] = gae
            episode[step]['return'] = gae + value
            next_value = value

    def rollout(self, train_ml=None, not_in_train=False, nss_w=0, **kwargs):
        obs = self.env._get_obs(t=0)
        batch_size = len(obs)
        lang_features, linear_cls, dialog_inputs = self.prepare_language_features(obs)

        current_view_corners = [np.array(ob['gt_path_corners'][0]) for ob in obs]
        current_directions = [np.array(ob['starting_angle']) for ob in obs]
        direction_t = torch.from_numpy(np.array(current_directions))
        traj = self.initialize_traj(obs, dialog_inputs)
        ended = np.array([False] * batch_size)
        ml_loss = torch.tensor(0.0, device=self.device)
        loss_logs = defaultdict(float)
        rollout_inputs = self.init_rollout_inputs(batch_size, lang_features, linear_cls)

        for t in range(self.args.max_action_len):
            im_feature = self.extract_visual_features(obs)
            rollout_inputs = self.append_step_inputs(rollout_inputs, im_feature, direction_t, ended)
            pred_dict, pred_saliency = self.forward_navigation(rollout_inputs)

            env_next_pos_ratio = self.normalize_waypoint(pred_dict['pred_next_pos_ratio'].detach().cpu().numpy())
            env_altitude = self.clip_altitude(pred_dict['pred_altitude'].detach().cpu().numpy())

            target = None
            gt_progress = None
            stop_label = None
            if 'test' not in self.env_name:
                target, gt_progress = self.teacher_action(obs, ended, current_view_corners, current_directions)
                il_loss, step_logs, stop_label = self.compute_il_losses(pred_dict, target, gt_progress, ended)
                saliency_loss, nss_loss = self.compute_saliency_loss(pred_saliency, obs, traj, not_in_train, nss_w, t)
                ml_loss = ml_loss + il_loss + saliency_loss
                for key, value in step_logs.items():
                    loss_logs[key] += value
                loss_logs['nss_loss'] += nss_loss

            stop_decision = self.should_stop(pred_dict['stop_prob'], stop_label)
            action_source = [[env_next_pos_ratio[j], env_altitude[j]] for j in range(len(obs))]
            if self.feedback == 'teacher' and target is not None:
                action_source = target

            for i in range(len(obs)):
                if ended[i]:
                    continue
                traj[i]['actions'].append([env_next_pos_ratio[i], env_altitude[i]])
                traj[i]['progress'].append(pred_dict['pred_progress'][i].item())
                traj[i]['stop_probs'].append(pred_dict['stop_prob'][i].item())
                traj[i]['stop_preds'].append(float(stop_decision[i]))
                if 'test' not in self.env_name:
                    traj[i]['gt_actions'].append(target[i])
                    traj[i]['gt_progress'].append(float(gt_progress[i][0]))
                    traj[i]['gt_stop'].append(float(stop_label[i].item()))

            for i in range(len(obs)):
                if ended[i]:
                    continue
                if stop_decision[i] or t == self.args.max_action_len - 1:
                    ended[i] = True
                    continue
                current_view_corners[i], current_directions[i] = self.execute_action(
                    current_view_corners[i], current_directions[i], action_source[i], obs[i]
                )

            for i in range(len(obs)):
                if not ended[i]:
                    traj[i]['path_corners'].append((current_view_corners[i], current_directions[i]))

            direction_t = torch.from_numpy(np.array(current_directions))
            obs = self.env._get_obs(corners=current_view_corners, directions=current_directions)
            if ended.all():
                break

        if self.args.inference and 'test' not in self.env_name:
            self.visualize_trajectory(obs, traj, not_in_train)

        if train_ml is not None:
            total_il_loss = ml_loss * train_ml / batch_size
            self.loss += total_il_loss
            self.logs['IL_loss'].append(total_il_loss.item())
            for key, value in loss_logs.items():
                self.logs[key].append(value * train_ml / max(batch_size, 1))

        if isinstance(self.loss, int):
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.args.max_action_len)
        return traj

    def visualize_trajectory(self, obs, traj, not_in_train):
        for i in range(len(obs)):
            img = self.env.map_batch[obs[i]['map_name']].copy()
            for j in range(len(traj[i]['actions'])):
                mean_coord = self.gps_to_img_coords(
                    np.mean(traj[i]['path_corners'][j][0], axis=0),
                    obs[i]['gps_botm_left'],
                    obs[i]['gps_top_right'],
                    obs[i]['lat_ratio'],
                )
                mean_coord = np.array(mean_coord, dtype=np.int32)
                a_direction = (math.atan2(traj[i]['actions'][j][0][0], traj[i]['actions'][j][0][1]) / 3.14159 + 2) / 2 % 1
                a_distance = np.linalg.norm(traj[i]['actions'][j][0]) * (np.linalg.norm(traj[i]['path_corners'][j][0][0] - traj[i]['path_corners'][j][0][1]) / 2)
                a_altitude = traj[i]['actions'][j][1]
                cv2.drawContours(img, [np.array(
                    [[self.gps_to_img_coords([traj[i]['path_corners'][j][0][0][0], traj[i]['path_corners'][j][0][0][1]], obs[i]['gps_botm_left'], obs[i]['gps_top_right'], obs[i]['lat_ratio'])],
                     [self.gps_to_img_coords([traj[i]['path_corners'][j][0][1][0], traj[i]['path_corners'][j][0][1][1]], obs[i]['gps_botm_left'], obs[i]['gps_top_right'], obs[i]['lat_ratio'])],
                     [self.gps_to_img_coords([traj[i]['path_corners'][j][0][2][0], traj[i]['path_corners'][j][0][2][1]], obs[i]['gps_botm_left'], obs[i]['gps_top_right'], obs[i]['lat_ratio'])],
                     [self.gps_to_img_coords([traj[i]['path_corners'][j][0][3][0], traj[i]['path_corners'][j][0][3][1]], obs[i]['gps_botm_left'], obs[i]['gps_top_right'], obs[i]['lat_ratio'])]])], 0, (255, 255, 255), 1)
                next_coord, _ = self.move_view_corners(
                    traj[i]['path_corners'][j][0],
                    round(a_direction * 360),
                    a_distance,
                    round(a_altitude * 360) + 40,
                    obs[i]['gps_botm_left'],
                    obs[i]['gps_top_right'],
                )
                next_coord = self.gps_to_img_coords(np.mean(next_coord, axis=0), obs[i]['gps_botm_left'], obs[i]['gps_top_right'], obs[i]['lat_ratio'])
                cv2.line(img, mean_coord, next_coord, (255, 0, 255), 4)
                cv2.circle(img, mean_coord, color=(255, 255, 255), thickness=2, radius=2)
                if j < len(traj[i].get('gt_actions', [])):
                    gt_action = traj[i]['gt_actions'][j]
                    cv2.putText(
                        img,
                        str(j) + ': [' + str(traj[i]['actions'][j][0][0])[:4] + ',' + str(traj[i]['actions'][j][0][1])[:4] +
                        '; ' + str(gt_action[0][0])[:4] + ',' + str(gt_action[0][1])[:4] + '] s=' +
                        str(traj[i]['stop_probs'][j])[:4] + ' p=' + str(traj[i]['progress'][j])[:4] + ', ' +
                        str(traj[i]['gt_progress'][j])[:4],
                        self.gps_to_img_coords([traj[i]['path_corners'][j][0][0][0], traj[i]['path_corners'][j][0][0][1]], obs[i]['gps_botm_left'], obs[i]['gps_top_right'], obs[i]['lat_ratio']),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA,
                    )
            cv2.putText(img, obs[i]['instructions'], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            if not_in_train:
                cv2.imwrite(self.args.pred_dir + '/debug_images/' + self.env_name + 'val' + obs[i]['map_name'] + '_' + obs[i]['route_index'] + '.jpg', img)

    def rollout_ppo(self):
        with torch.no_grad():
            obs = self.env._get_obs(t=0)
            batch_size = len(obs)
            lang_features, linear_cls, _ = self.prepare_language_features(obs)
            lang_features = lang_features.detach()
            linear_cls = linear_cls.detach()

            current_view_corners = [np.array(ob['gt_path_corners'][0]) for ob in obs]
            current_directions = [np.array(ob['starting_angle']) for ob in obs]
            direction_t = torch.from_numpy(np.array(current_directions))
            rollout_inputs = self.init_rollout_inputs(batch_size, lang_features, linear_cls)
            ended = np.array([False] * batch_size)
            episodes = [[] for _ in range(batch_size)]
            episode_rewards = [[] for _ in range(batch_size)]

            for t in range(self.args.max_action_len):
                im_feature = self.extract_visual_features(obs).detach()
                rollout_inputs = self.append_step_inputs(rollout_inputs, im_feature, direction_t, ended)
                pred_dict, _ = self.forward_navigation(rollout_inputs)

                action_mean = torch.cat((pred_dict['pred_next_pos_ratio'], pred_dict['pred_altitude'].unsqueeze(1)), dim=1)
                action_std = pred_dict['action_log_std'].exp().unsqueeze(0).expand_as(action_mean)
                action_dist = Normal(action_mean, action_std)
                stop_dist = Bernoulli(logits=pred_dict['stop_logit'])

                sampled_action = action_dist.sample()
                sampled_stop = stop_dist.sample()
                old_log_prob = action_dist.log_prob(sampled_action).sum(dim=1) + stop_dist.log_prob(sampled_stop)
                env_next_pos_ratio, env_altitude = self.policy_action_to_env_action(sampled_action)

                _, gt_progress = self.teacher_action(obs, ended, current_view_corners, current_directions)
                gt_progress = np.asarray(gt_progress, dtype=np.float32).reshape(-1)
                teacher_stop = self.build_stop_label(gt_progress).detach().cpu().numpy()

                for i in range(batch_size):
                    if ended[i]:
                        continue
                    prev_iou = float(gt_progress[i])
                    transition = self.snapshot_transition(rollout_inputs, i)
                    transition.update({
                        'sampled_action': sampled_action[i].detach().cpu(),
                        'sampled_stop': float(sampled_stop[i].item()),
                        'old_log_prob': float(old_log_prob[i].item()),
                        'value': float(pred_dict['state_value'][i].item()),
                        'pred_progress': float(pred_dict['pred_progress'][i].item()),
                        'teacher_stop': float(teacher_stop[i]),
                    })

                    stop_action = bool(sampled_stop[i].item() >= 0.5)
                    timeout = t == self.args.max_action_len - 1
                    if stop_action:
                        current_iou = prev_iou
                    else:
                        current_view_corners[i], current_directions[i] = self.execute_action(
                            current_view_corners[i], current_directions[i], [env_next_pos_ratio[i], env_altitude[i]], obs[i]
                        )
                        current_iou = compute_iou(current_view_corners[i], obs[i]['gt_path_corners'][-1])

                    success_stop = stop_action and current_iou >= self.args.stop_iou_th
                    reward = self.compute_reward(prev_iou, current_iou, stop_action, timeout, success_stop)
                    done = stop_action or timeout

                    transition['reward'] = float(reward)
                    transition['done'] = bool(done)
                    episodes[i].append(transition)
                    episode_rewards[i].append(float(reward))
                    if done:
                        ended[i] = True

                direction_t = torch.from_numpy(np.array(current_directions))
                obs = self.env._get_obs(corners=current_view_corners, directions=current_directions)
                if ended.all():
                    break

        flat_transitions = []
        for episode in episodes:
            if not episode:
                continue
            self.compute_episode_gae(episode)
            flat_transitions.extend(episode)

        mean_return = float(np.mean([np.sum(rews) for rews in episode_rewards if rews])) if any(len(rews) > 0 for rews in episode_rewards) else 0.0
        mean_len = float(np.mean([len(ep) for ep in episodes if ep])) if any(len(ep) > 0 for ep in episodes) else 0.0
        return flat_transitions, {'episodes': len([ep for ep in episodes if ep]), 'mean_return': mean_return, 'mean_len': mean_len}

    def update_ppo(self, transitions):
        if len(transitions) == 0:
            return {}

        advantages = torch.tensor([tr['advantage'] for tr in transitions], dtype=torch.float32, device=self.device)
        returns = torch.tensor([tr['return'] for tr in transitions], dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor([tr['old_log_prob'] for tr in transitions], dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        stats = defaultdict(list)
        indices = np.arange(len(transitions))
        minibatch_size = min(self.args.ppo_minibatch_size, len(transitions))

        self.vln_model.eval()
        for _ in range(self.args.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), minibatch_size):
                batch_idx = indices[start:start + minibatch_size]
                self.rl_optimizer.zero_grad()
                total_loss = torch.tensor(0.0, device=self.device)
                policy_losses = []
                value_losses = []
                entropies = []
                ratios = []

                for idx in batch_idx:
                    log_prob, value, entropy = self.recompute_policy_metrics(transitions[idx])
                    ratio = torch.exp(log_prob - old_log_probs[idx])
                    surr1 = ratio * advantages[idx]
                    surr2 = torch.clamp(ratio, 1.0 - self.args.ppo_clip, 1.0 + self.args.ppo_clip) * advantages[idx]
                    policy_loss = -torch.min(surr1, surr2)
                    value_loss = F.mse_loss(value, returns[idx], reduction='mean')
                    loss = policy_loss + self.args.ppo_value_w * value_loss - self.args.ppo_entropy_w * entropy
                    total_loss = total_loss + loss

                    policy_losses.append(policy_loss.detach().item())
                    value_losses.append(value_loss.detach().item())
                    entropies.append(entropy.detach().item())
                    ratios.append(ratio.detach().item())

                total_loss = total_loss / max(len(batch_idx), 1)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vln_model.parameters(), 10.)
                self.rl_optimizer.step()

                stats['ppo_loss'].append(float(total_loss.detach().item()))
                stats['ppo_policy_loss'].append(float(np.mean(policy_losses)) if policy_losses else 0.0)
                stats['ppo_value_loss'].append(float(np.mean(value_losses)) if value_losses else 0.0)
                stats['ppo_entropy'].append(float(np.mean(entropies)) if entropies else 0.0)
                stats['ppo_ratio'].append(float(np.mean(ratios)) if ratios else 1.0)
        return {k: float(np.mean(v)) for k, v in stats.items()}

    def run_ppo_il_anchor(self):
        if self.args.ppo_il_w <= 0:
            return {}
        self.rl_optimizer.zero_grad()
        self.lang_model_optimizer.zero_grad()
        self.vision_model_optimizer.zero_grad()
        self.loss = 0
        current_feedback = self.feedback
        self.feedback = 'teacher'
        self.rollout(train_ml=self.args.ppo_il_w, train_rl=False, nss_w=0)
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vln_model.parameters(), 10.)
        self.rl_optimizer.step()
        self.lang_model_optimizer.zero_grad()
        self.vision_model_optimizer.zero_grad()
        self.feedback = current_feedback
        return {'ppo_il_anchor': float(self.loss.detach().item()) if not isinstance(self.loss, int) else 0.0}

    def save(self, epoch, path):
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}

        def create_state(name, model, optimizer):
            model = self._unwrap(model)
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

        create_state('lang_model', self.lang_model, self.lang_model_optimizer)
        create_state('vision_model', self.vision_model, self.vision_model_optimizer)
        create_state('vln_model', self.vln_model, self.et_optimizer)
        states['rl_optimizer'] = {'epoch': epoch + 1, 'optimizer': self.rl_optimizer.state_dict()}
        torch.save(states, path)

    def load(self, path):
        states = torch.load(path, map_location='cpu')

        def recover_state(name, model, optimizer):
            model = self._unwrap(model)
            state = model.state_dict()
            model_keys = set(state.keys())
            if name in states:
                load_state = states[name]['state_dict']
                load_keys = set(load_state.keys())
                if model_keys == load_keys:
                    print('NOTICE: LOADing ALL KEYS IN THE ', name)
                    state_dict = load_state
                else:
                    print('NOTICE: DIFFERENT KEYS IN THE ', name)
                    state_dict = {k: v for k, v in load_state.items() if k in model_keys}
                state.update(state_dict)
                model.load_state_dict(state)
                if self.args.resume_optimizer and 'optimizer' in states[name]:
                    optimizer.load_state_dict(states[name]['optimizer'])
            count_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print('Model parameters: ', count_parameters)

        recover_state('lang_model', self.lang_model, self.lang_model_optimizer)
        recover_state('vision_model', self.vision_model, self.vision_model_optimizer)
        recover_state('vln_model', self.vln_model, self.et_optimizer)

        if self.args.resume_optimizer and 'rl_optimizer' in states and 'optimizer' in states['rl_optimizer']:
            self.rl_optimizer.load_state_dict(states['rl_optimizer']['optimizer'])
        return states['vln_model']['epoch'] - 1
