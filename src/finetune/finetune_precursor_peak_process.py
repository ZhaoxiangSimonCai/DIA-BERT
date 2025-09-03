import concurrent
import math
import os
import pickle
import shutil
from sklearn.model_selection import StratifiedKFold
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import runtime_data_info
from src.common.constant import ProgressStepEnum, ProgressStepStatusEnum
from src.common.model.finetune_dataset import Dataset
from src.common.model.score_model import FeatureEngineer
from src.utils import msg_send_utils
from src import common_config

common_config_data = common_config.read_yml()

class FinetunePrecursorPeakProcess(object):

    def __init__(self, mzml_name=None, base_output=None, n_thread=1, score_limit=0.5, each_pkl_size=6144, rt_index=None, instrument_index=None, lib_max_intensity=None, logger=None):
        self.mzml_name = mzml_name
        self.base_output = base_output
        self.pkl_dir = os.path.join(self.base_output, 'identify_data')

        self.n_thread = n_thread
        self.score_limit = score_limit
        self.each_pkl_size = each_pkl_size

        self.rt_index = rt_index
        self.instrument_index = instrument_index
        self.lib_max_intensity = lib_max_intensity

        self.logger = logger

    def peak_score_precursor(self):
        msg_send_utils.send_msg(step=ProgressStepEnum.PREPARE_DATA, status=ProgressStepStatusEnum.RUNNING,
                                msg='Processing prepare data, score_limit: {}'.format(self.score_limit))
        try:
            if not runtime_data_info.runtime_data.current_is_success:
                msg_send_utils.send_msg(step=ProgressStepEnum.PREPARE_DATA, status=ProgressStepStatusEnum.ERROR)
                return
            self.peak_score()
            self.get_prophet_result()
            #
            self.peak_rsm()
            #
            # self.split_pkl()
        except Exception:
            self.logger.exception('Finetune prepare data exception')
            runtime_data_info.runtime_data.current_is_success = False
            msg_send_utils.send_msg(step=ProgressStepEnum.PREPARE_DATA, status=ProgressStepStatusEnum.ERROR)

    def get_prophet_result(self):
        self.logger.info('Processing calc score more {} precursor'.format(self.score_limit))
        msg_send_utils.send_msg(msg='Calc score more {} precursor'.format(self.score_limit))
        fdr_dir = os.path.join(self.base_output, 'fdr_stats')
        file_path = os.path.join(fdr_dir, "{}_precursor.tsv".format(self.mzml_name))
        df = pd.read_table(file_path)

        df.columns = ['transition_group_id', 'score', 'label', 'file_name']
        df['score'] = df['score'].astype(float)
        df['label'] = df['label'].astype(int)
        df = df.sort_values(by='score', ascending=False, ignore_index=True)
        df['decoy'] = np.where(df['label'] == 1, 0, 1)

        target_num = (df.decoy == 0).cumsum()
        decoy_num = (df.decoy == 1).cumsum()

        target_num[target_num == 0] = 1
        decoy_num[decoy_num == 0] = 1
        df['q_value'] = decoy_num / target_num
        df['q_value'] = df['q_value'][::-1].cummin()

        target = df[(df['score'] >= self.score_limit) & (df['label'] == 1)]
        decoy = df[(df['label'] == 0)].head(len(target))

        # filtered_df = df[df['transition_group_id'].isin(set(target['transition_group_id']) | set(decoy['transition_group_id']))][['transition_group_id', 'score', 'label', 'q_value', 'file_name']]

        usecols= ['transition_group_id', 'score', 'label', 'file_name']
        filtered_df = pd.concat([target[usecols], decoy[usecols]], axis=0)

        filtered_df = self.get_KFold(filtered_df)
        csv_path = os.path.join(self.base_output, 'fdr_stats', '{}_score_{}_precursor.csv'.format(self.mzml_name, self.score_limit))
        filtered_df.to_csv(csv_path, index=False)
        self.logger.info('Finish calc score more {} precursor, there have {} records, save info to {}'.format(self.score_limit, len(target) * 2, csv_path))
        msg_send_utils.send_msg(msg='Finish calc score more {} precursor, there have {} records, save info to {}'.format(self.score_limit, len(target) * 2, csv_path))

    def get_KFold(self, df, group_key='transition_group_id', label_col='label'):
        # 按照 transition_group_id 分组，确保每个 transition_group_id 只会出现在一折中
        grouped = df.groupby(group_key).first().reset_index()

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # 创建一个新的列 fold 来存储每折的索引
        grouped["fold"] = -1

        # 使用 StratifiedKFold 对 transition_group_id 进行分层划分
        for fold, (train_idx, val_idx) in enumerate(skf.split(grouped, grouped[label_col])):
            grouped.loc[val_idx, "fold"] = fold

        # 将分组后的 fold 映射回原始 DataFrame
        df = df.merge(grouped[[group_key, "fold"]], on=group_key, how="left")
        return df

    def construct_data_set(self, pkl_data_list):
        file_diabert_score = []
        file_precursor_id = []
        file_target = []
        for pkl_name in pkl_data_list:
            pkl_path = os.path.join(self.pkl_dir, 'score', pkl_name)
            with open(pkl_path, mode='rb') as f:
                precursor, score = pickle.load(f)
                precursor_np = np.array(precursor)
                file_diabert_score.extend(score.tolist())
                file_precursor_id.extend(precursor_np[:, 0].tolist())
                file_target.extend([1 - int(x) for x in precursor_np[:, 1].tolist()])

        return file_precursor_id, file_target, file_diabert_score

    '''
    
    '''
    def peak_score(self):
        self.logger.info('Processing peak score info')
        msg_send_utils.send_msg(msg='Processing peak score info')
        score_info_dir = os.path.join(self.pkl_dir, 'score')
        pkl_list = os.listdir(score_info_dir)
        pkl_list = list(filter(lambda entry: entry.endswith('.pkl'), pkl_list))
        pkl_list = sorted(pkl_list)

        precursor_id_list, target_flag_list, score_list = self.construct_data_set(pkl_list)

        fdr_dir = os.path.join(self.base_output, 'fdr_stats')
        if os.path.exists(fdr_dir):
            shutil.rmtree(fdr_dir)
        os.makedirs(fdr_dir)

        df = pd.DataFrame(score_list, index=precursor_id_list, columns=['score'])
        df['score'] = df['score'].astype(np.float32)
        df["target"] = target_flag_list
        df["filename"] = self.mzml_name
        file_path = os.path.join(fdr_dir, "{}_precursor.tsv".format(self.mzml_name))
        df.to_csv(file_path, sep="\t")
        self.logger.info('Finish peak score info, there have {} precursor'.format(len(precursor_id_list)))
        msg_send_utils.send_msg(msg='Finish peak score info, there have {} precursor'.format(len(precursor_id_list)))

    '''
    
    '''
    def peak_rsm(self):
        self.logger.info('Processing peak precursor rsm info for finetune')
        msg_send_utils.send_msg(msg='Peak precursor rsm info for finetune')
        finetune_dir = os.path.join(self.base_output, 'finetune')
        if os.path.exists(finetune_dir):
            shutil.rmtree(finetune_dir)
        finetune_pkl_dir = os.path.join(finetune_dir, 'data')
        os.makedirs(finetune_pkl_dir)
        #
        csv_path = os.path.join(self.base_output, 'fdr_stats', '{}_score_{}_precursor.csv'.format(self.mzml_name, self.score_limit))
        score_more_df = pd.read_csv(csv_path)

        # 获取need_precursor和fold的对应关系
        need_precursor_fold_dict = score_more_df.set_index('transition_group_id')['fold'].to_dict()

        pkl_list = os.listdir(self.pkl_dir)
        pkl_list = list(filter(lambda entry: entry.endswith('.pkl'), pkl_list))
        pkl_list = sorted(pkl_list)
        #
        # precursor_id_list, rsm_list, frag_info_list, precursor_feat_list, target_info_list = [], [], [], [], []
        feature_engineer = FeatureEngineer()
        # save_pkl_index = 0


        fold_rsm_arr = [[] for _ in range(5)]
        fold_feat_arr = [[] for _ in range(5)]
        fold_frag_info_arr = [[] for _ in range(5)]
        target_info_arr = [[] for _ in range(5)]
        precursor_id_list_arr = [[] for _ in range(5)]

        for pkl_name in pkl_list:
            pkl_path = os.path.join(self.pkl_dir, pkl_name)
            with open(pkl_path, mode='rb') as f:
                precursor, precursor_feat, rsm, frag_info, score = pickle.load(f)

                precursor_np = np.array(precursor)
                pkl_precursor_id_list = precursor_np[:, 0].tolist()

                fold1_peak_index_list = [index for index, precursor in enumerate(pkl_precursor_id_list) if
                                         need_precursor_fold_dict.get(precursor) == 0]
                fold2_peak_index_list = [index for index, precursor in enumerate(pkl_precursor_id_list) if
                                         need_precursor_fold_dict.get(precursor) == 1]
                fold3_peak_index_list = [index for index, precursor in enumerate(pkl_precursor_id_list) if
                                         need_precursor_fold_dict.get(precursor) == 2]
                fold4_peak_index_list = [index for index, precursor in enumerate(pkl_precursor_id_list) if
                                         need_precursor_fold_dict.get(precursor) == 3]
                fold5_peak_index_list = [index for index, precursor in enumerate(pkl_precursor_id_list) if
                                         need_precursor_fold_dict.get(precursor) == 4]

                peak_index_list_arr = [fold1_peak_index_list, fold2_peak_index_list, fold3_peak_index_list, fold4_peak_index_list, fold5_peak_index_list]

                for fold_index, peak_index_list in enumerate(peak_index_list_arr):
                    if len(peak_index_list) == 0:
                        continue
                    precursor_id = [pkl_precursor_id_list[index] for index in peak_index_list]
                    fold_rsm_arr[fold_index].append(rsm[peak_index_list])
                    fold_feat_arr[fold_index].append(precursor_feat[peak_index_list])
                    fold_frag_info_arr[fold_index].append(frag_info[peak_index_list])
                    precursor_id_list_arr[fold_index].extend(precursor_id)
                    target_info_arr[fold_index].extend([1 - int(x) for x in precursor_np[:, 1][peak_index_list].tolist()])

        # 循环处理
        for fold_index, precursor_id_list in enumerate(precursor_id_list_arr):
            rsm_list = fold_rsm_arr[fold_index]
            frag_info_list = fold_frag_info_arr[fold_index]
            precursor_feat_list = fold_feat_arr[fold_index]
            target_info_list = target_info_arr[fold_index]

            all_rsm = np.concatenate(rsm_list, axis=0)
            all_frag_info = np.concatenate(frag_info_list, axis=0)
            all_precursor_feat = np.concatenate(precursor_feat_list, axis=0)
            save_rsm, rsm_max = FeatureEngineer.process_intensity_np(all_rsm)
            save_rsm = save_rsm.swapaxes(1, 2)
            save_frag_info = feature_engineer.process_frag_info(all_frag_info, max_intensity=self.lib_max_intensity)
            save_precursor_feat = np.column_stack((all_precursor_feat[:, :7], rsm_max))
            save_precursor_feat = feature_engineer.process_feat_np(save_precursor_feat)
            #
            pr_ids = len(precursor_id_list)
            rt_np = np.array([self.rt_index]).repeat(pr_ids).reshape(-1, 1)
            instrument_np = np.array([self.instrument_index]).repeat(pr_ids).reshape(-1, 1)
            save_precursor_feat = np.concatenate((save_precursor_feat, rt_np, instrument_np), axis=1)
            data_set = Dataset()
            data_set.rsm = save_rsm
            data_set.feat = save_precursor_feat
            data_set.frag_info = save_frag_info
            data_set.label = target_info_list
            data_set.precursor_id = precursor_id_list
            data_set.file = [self.mzml_name for _ in range(len(precursor_id_list))]
            with open(os.path.join(finetune_pkl_dir, f'chunk_{fold_index}.pkl'), mode='wb') as f:
                f.write(pickle.dumps(data_set, protocol=4))
        self.logger.info('Finished peak precursor rsm info for finetune')
        msg_send_utils.send_msg(msg='Finished peak precursor rsm info for finetune')

    '''
    
    '''
    def split_pkl(self):
        self.logger.info('Split test and train data')
        msg_send_utils.send_msg(msg='Split test and train data')
        #
        finetune_pkl_dir = os.path.join(self.base_output, 'finetune', 'data')
        pkl_list = os.listdir(finetune_pkl_dir)

        pkl_list = list(filter(lambda entry: entry.endswith('.pkl'), pkl_list))
        if len(pkl_list) == 0:
            self.logger.error('The finetune pkl is none')
            msg_send_utils.send_msg(step=ProgressStepEnum.PREPARE_DATA, status=ProgressStepStatusEnum.ERROR, msg='The finetune pkl is none')
            return

        if len(pkl_list) < 5:
            self.logger.error('The finetune pkl is not enough, min need 5, but have {}'.format(len(pkl_list)))
            msg_send_utils.send_msg(step=ProgressStepEnum.PREPARE_DATA, status=ProgressStepStatusEnum.ERROR, msg='The finetune pkl is not enough, min need 5, but have {}'.format(len(pkl_list)))
            return

        train_path_list, test_path_list = train_test_split(pkl_list, random_state=123, test_size=0.2)
        train_pkl_dir = os.path.join(finetune_pkl_dir, 'sp_train_feat')
        test_pkl_dir = os.path.join(finetune_pkl_dir, 'sp_test_feat')
        if not os.path.exists(train_pkl_dir):
            os.makedirs(train_pkl_dir)
        if not os.path.exists(test_pkl_dir):
            os.makedirs(test_pkl_dir)
        #
        for pkl_name in train_path_list:
            shutil.move(os.path.join(finetune_pkl_dir, pkl_name), os.path.join(train_pkl_dir, pkl_name))
        for pkl_name in test_path_list:
            shutil.move(os.path.join(finetune_pkl_dir, pkl_name), os.path.join(test_pkl_dir, pkl_name))
        self.logger.info('Finish split test and train data')
        msg_send_utils.send_msg(step=ProgressStepEnum.PREPARE_DATA, status=ProgressStepStatusEnum.SUCCESS, msg='Finish split test and train data')
