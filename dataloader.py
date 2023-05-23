from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random
from six.moves import cPickle
import pickle

import torch
import torch.utils.data as data
#import ipdb

import multiprocessing


def zero_pad(features,n_feat):
    if features.shape[0] < n_feat:
        features = np.vstack((features,np.zeros((n_feat - features.shape[0], features.shape[1]))))
    return features

# https://stackoverflow.com/questions/25200220/generate-a-random-derangement-of-a-list
def random_derangement(n):
    if n == 0 or n == 1:
        return n
    while True:
        v = range(n)
        for j in range(n - 1, -1, -1):
            p = random.randint(0, j)
            if v[p] == j:
                break
            else:
                v[j], v[p] = v[p], v[j]
        else:
            if v[0] != 0:
                return v

class DataLoader(data.Dataset):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_wtoi(self):
        return self.word_to_ix

    def get_seq_length(self):
        return self.seq_length

    def get_blank_token(self):
        return self.word_to_ix['<blank>']

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size

        self.input_fc_dir = self.opt.input_fc_dir
        self.use_video = self.opt.use_video
        self.input_img_dir = self.opt.input_img_dir
        self.use_img = self.opt.use_img
        self.input_face_dir = self.opt.input_face_dir

        # New DataLoader Feature Variables
        self.mean_pool_video = True
        self.max_videos_in_group = 5


        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']
        self.word_to_ix = self.info['word_to_ix']
        self.groups = self.info['groups']
        self.movie_dict = self.info['movie_ids']
        self.seq_length = self.info['max_seq_length']
        self.max_caption_length = 120
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)
        print('max sequence length in data is', self.seq_length)

        self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r')
        self.captions = self.h5_label_file['labels'].value
        self.max_characters = self.info['max_character_count'] # 17
        self.unique_characters = self.info['unique_character_count'] # 11
        self.max_sent_num = self.opt.max_sent_num
        self.max_face = self.opt.max_face
        self.max_seg = self.opt.max_seg
        self.classify_gender = self.opt.classify_gender
        if self.classify_gender:
            self.clip_gender = json.load(open(self.opt.clip_gender_json))

        self.use_bert_embedding = 'use_bert_embedding' in vars(self.opt) and self.opt.use_bert_embedding
        self.bert_size = self.opt.bert_size  # 768 * 2
        if self.use_bert_embedding:
            self.bert_embedding_dir = self.opt.bert_embedding_dir
            print(f"bert_embedding_dir : {self.bert_embedding_dir}")
            self.bert_embedding = {'train' : pickle.load(open(os.path.join(self.bert_embedding_dir,'train_embeddings.pkl'),'rb')),
                            'val' : pickle.load(open(os.path.join(self.bert_embedding_dir, 'val_embeddings.pkl'),'rb')),
                            'test' : pickle.load(open(os.path.join(self.bert_embedding_dir, 'test_embeddings.pkl'),'rb'))}


        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': [], 'overfit': [], 'train_val' : []}
        self.split_size = {'train': 0, 'val': 0, 'test': 0, 'overfit': 0, 'train_val' : 0}
        self.split_map = {'train' : {}, 'val' : {}, 'test' : {}, 'overfit': {}, 'train_val': {}}
        self.ix_split = {}
        for j, group in enumerate(self.groups):
            id_val = group['id']
            if(id_val < 64):
                split = "overfit"
                self.split_ix[split].append(j)
                self.split_map[split][j] = self.split_size[split]
                self.split_size[split]+=1
                self.ix_split[j] = split
            if(id_val < 1402):
                split = "train_val"
                self.split_ix[split].append(j)
                self.split_map[split][j] = self.split_size[split]
                self.split_size[split]+=1
                self.ix_split[j] = split
            split = group['split']
            self.split_ix[split].append(j)
            self.split_map[split][j] = self.split_size[split]
            self.split_size[split]+=1
            self.ix_split[j] = split

        print('assigned %d videos to split train' % len(self.split_ix['train']))
        print('assigned %d videos to split train_val' % len(self.split_ix['train_val']))
        print('assigned %d videos to split val' % len(self.split_ix['val']))
        print('assigned %d videos to split test' % len(self.split_ix['test']))
        print('assigned %d videos to split overfit' % len(self.split_ix['overfit']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0, 'overfit': 0, 'train_val':0}

        self._prefetch_process = {} # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
            # Terminate the child process when the parent exists
        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]
        import atexit
        atexit.register(cleanup)

    # mean pool the features across max_seg segments
    def meanpool_segments(self, features):
        if features.shape[0] >= self.max_seg:
            tmp_feat = []
            # self.max_seg - T - 5
            nps = int(np.floor(features.shape[0] // self.max_seg))  # numbers per segment
            for i in range(self.max_seg):
                if i != self.max_seg - 1:
                    segment = features[nps * i:nps * (i + 1)]
                else:
                    segment = features[nps * i:]
                segment = segment.mean(axis=0)
                tmp_feat.append(segment)
            features = np.array(tmp_feat)
        else:
            # 0 pad frames
            features = zero_pad(features, self.max_seg)
        # features will be --> 5x1024
        return features

    def get_sent_num(self, index):
        return len(self.groups[index]['videos'])

    def get_slot_batch(self, index):
        """
        :param index:
        :return: sent_ix =  array of indices of clips for each slot (len is # of total characters in video)
                 sent_num = # of clips for the video index
        """
        v_idx = self.groups[index]['videos']
        slots = []
        for i, id in enumerate(v_idx):
            n_blanks = self.info['videos'][id]['num_blanks']
            if n_blanks > 0:
                for _ in range(n_blanks):
                    slots.append(i)
        return slots

    def get_caption_batch(self, index):
        indexed_caption = self.groups[index]['index_caption']
        np_caption  = np.array(indexed_caption[:self.max_caption_length])
        caption_mask = np.ones_like(np_caption)
        if(len(np_caption) == self.max_caption_length):
            np_caption[-1] = 102
        np_caption = np.pad(np_caption, (0, self.max_caption_length - len(np_caption)), 'constant')
        caption_mask = np.pad(caption_mask, (0, self.max_caption_length - len(caption_mask)), 'constant')
        return np_caption, caption_mask
    
    def get_gt_caption_batch(self, index):
        indexed_gt_caption = self.groups[index]['index_gt_caption']
        np_gt_caption  = np.array(indexed_gt_caption[:self.max_caption_length])
        np_caption_mask = np.ones_like(np_gt_caption)
        if(len(np_gt_caption) == self.max_caption_length):
            np_gt_caption[-1] = 30537
        np_gt_caption = np.pad(np_gt_caption, (0, self.max_caption_length - len(np_gt_caption)), 'constant')
        np_caption_mask = np.pad(np_caption_mask, (0, self.max_caption_length - len(np_caption_mask)), 'constant')
        return np_gt_caption, np_caption_mask

    def get_character_batch(self,index):
        v_idx = self.groups[index]['videos']
        character_ids = []
        for id in v_idx:
            n_blanks = self.info['videos'][id]['num_blanks']
            if n_blanks > 0:
                characters = self.info['videos'][id]['character_id']
                for n in range(n_blanks):
                    character_ids.append(characters[n])

        character_map = {}
        for c in character_ids:
            if c not in character_map:
                character_map[c] = len(character_map.keys()) + 1
        character_ids = [character_map[c] for c in character_ids]
        return character_ids

    def get_bert_batch(self,index, split):
        main_split = split
        # 'overfit': [], 'train_val'
        if(split == "overfit" or split == "train_val"):
            main_split = "train"
        return self.bert_embedding[main_split][self.split_map[split][index]]
    
    def get_caption_segment_batch(self, index):
        position_ids = self.groups[index]['position_ids']
        segment_ids = self.groups[index]['segment_ids']
        blank_indexes = self.groups[index]['blank_indexes']
        blank_masks = self.groups[index]['blank_masks']

        return position_ids, segment_ids, blank_indexes, blank_masks

    def get_gender_batch(self,index):
        gender_map = {'M' : 0, 'F' : 1, 'G' : 2}
        v_idx = self.groups[index]['videos']
        gender_ids = []

        for id in v_idx:
            clip = self.info['videos'][id]['clip']
            n_blanks = self.info['videos'][id]['num_blanks']
            if n_blanks > 0:
                for n in range(n_blanks):
                    gender_ids.append(gender_map[self.clip_gender[clip][n]])
        return gender_ids

    def get_seg_batch(self, index):
        """
        :param index: group index in batch
        :return: fc_features   [sent_num x D_fc]
                 face_features [sent_num x face_num x D_ff]
        """
        v_idx = self.groups[index]['videos']
        sent_num = len(v_idx)
        assert sent_num > 0, 'data should have at least one caption'
        fc_features = []
        face_features = []
        face_masks = []
        face_segment_ids = []

        for id in v_idx:
            n_blanks = self.info['videos'][id]['num_blanks']
            movie = self.info['videos'][id]['movie']
            clip = self.info['videos'][id]['clip']

        
            fc_dir = [self.input_fc_dir, movie, clip + '.npy']
            fc_feats = np.load(os.path.join(*fc_dir))
            fc_max_seg = min(fc_feats.shape[0], self.max_seg)
            if(self.mean_pool_video):
                fc_feats = self.meanpool_segments(fc_feats)

            face_dir = [self.input_face_dir, movie, clip + '.npy']
            face_npy = np.load(os.path.join(*face_dir))
            face_mask = np.zeros(self.max_face, dtype = 'float32')
            # self.max_face -- 10
            # Padding the face cluster features with zeros.
            if face_npy.shape[0] > 0:
                face_feats = zero_pad(face_npy, self.max_face)
                face_mask[:min(face_npy.shape[0], self.max_face)] = 1
            else:
                face_feats = np.zeros([self.max_face, self.opt.face_feat_size], dtype = 'float32')
            
            face_feats = face_feats[:self.max_face]

            # Find the face segment id
            temporal_dist = face_feats[:,1]
            fc_idx = (fc_max_seg * temporal_dist - 1e-6).astype(int)
            # Pick only the face features.
            face_feats = face_feats[:,6:] 

            '''
            for i in range(len(fc_idx) -1 , -1, -1):
                if(fc_idx[i] != 0):
                    break
                else:
                    fc_idx[i] = -1
            '''

            fc_features.append(fc_feats) # Shape would be 5x1024 now.
            face_features.append(face_feats)
            face_masks.append(face_mask)
            face_segment_ids.append(fc_idx)
                
                
        return np.array(fc_features), np.array(face_features), np.array(face_masks), np.array(face_segment_ids)

    # Each batch is a video with multiple clips/sentences
    def get_batch(self, split, batch_size=None):
        batch_size = batch_size or self.batch_size

        # inputs for training
        slot_batch = np.ones((batch_size, self.max_characters), dtype='int') * -1
        sent_num_batch = np.zeros(batch_size, dtype = 'int')
        fc_batch = np.zeros([batch_size, self.max_videos_in_group, self.max_seg, self.opt.fc_feat_size], dtype = 'float32')
        face_batch = np.zeros([batch_size, self.max_videos_in_group, self.max_face, self.opt.face_feat_size - 6], dtype = 'float32')
        face_mask_batch = np.zeros([batch_size, self.max_videos_in_group, self.max_face], dtype = 'float32')

        bert_batch = np.zeros((batch_size, self.max_characters, self.bert_size), dtype='float32')
        caption_batch = np.zeros((batch_size, self.max_caption_length), dtype='int')
        mask_batch = np.zeros((batch_size, self.max_caption_length), dtype='float32')
        caption_gt_batch = np.zeros((batch_size, self.max_caption_length), dtype='int')
        caption_gt_mask = np.zeros((batch_size, self.max_caption_length), dtype='float32')

        gender_batch = np.zeros((batch_size, self.max_characters),dtype='int') if self.classify_gender else None
        character_batch = np.zeros((batch_size, self.max_characters+1), dtype='int')
        slot_mask_batch = np.zeros((batch_size, self.max_characters+1), dtype='int')

        face_segment_batch = np.zeros((batch_size, self.max_videos_in_group, self.max_face), dtype='int')

        # Caption Positions and Segments
        caption_position_batch = np.zeros((batch_size, self.max_caption_length), dtype='int')
        caption_segment_batch = np.zeros((batch_size, self.max_caption_length), dtype='int')
        blank_indexes_batch = np.zeros((batch_size, self.max_characters), dtype='int')
        blank_masks_batch = np.zeros((batch_size, self.max_caption_length), dtype='int')

        wrapped = False
        infos = []
        slot_size = []
        i = 0
        while i < batch_size:
            # fetch visual features
            tmp_fcs, ix, tmp_wrapped = self._prefetch_process[split].get()
            slots = self.get_slot_batch(ix)
            slot_num = len(slots)
            slot_size.append(slot_num)

            if tmp_wrapped:
                wrapped = True

            vids_in_group = len(self.groups[ix]['videos'])
            print(f"Index is {ix}")
            
            for v_ix in self.groups[ix]['videos']:
                info_dict = {}
                info_dict['index'] = v_ix
                info_dict['g_index'] = ix
                info_dict['id'] = self.info['videos'][v_ix]['clip']
                info_dict['caption'] = self.captions[v_ix]
                info_dict['skipped'] = slot_num==0
                infos.append(info_dict)

            if slot_num == 0:
                continue
            else:

                if split != 'test':
                    if self.classify_gender:
                        gender_batch[i, :slot_num] = self.get_gender_batch(ix)
                    character = self.get_character_batch(ix)
                    character_batch[i,1:slot_num+1] = character
                    caption_gt_batch[i], caption_gt_mask[i] = self.get_gt_caption_batch(ix)
                    
                
                fc_batch[i,:vids_in_group,:,:] = tmp_fcs[0]
                face_batch[i,:vids_in_group,:,:] = tmp_fcs[1]
                face_mask_batch[i,:vids_in_group,:] = tmp_fcs[2]
                face_segment_batch[i,:vids_in_group,:] = tmp_fcs[3]

                if self.use_bert_embedding:
                    bert_batch[i] = self.get_bert_batch(ix, split)
                caption_batch[i], mask_batch[i] = self.get_caption_batch(ix)

                position_ids, segment_ids, blank_indexes, blank_masks = self.get_caption_segment_batch(ix)
                #caption_position_batch[i], caption_segment_batch[i], blank_indexes_batch[i] = self.get_caption_segment_batch(ix)
                max_pos_id_length = min(self.max_caption_length, len(position_ids))
                max_seg_id_length = min(self.max_caption_length, len(segment_ids))
                caption_position_batch[i, :max_pos_id_length] = position_ids[:max_pos_id_length]
                caption_segment_batch[i, :max_seg_id_length] = segment_ids[:max_seg_id_length]
                blank_indexes_batch[i, :len(blank_indexes)] = blank_indexes
                max_blank_mask_length = min(self.max_caption_length, len(blank_masks))
                blank_masks_batch[i, :max_blank_mask_length] = blank_masks[:max_blank_mask_length]

                sent_num = self.get_sent_num(ix)
                sent_num_batch[i] = sent_num
                slot_batch[i,:slot_num] = slots
                slot_mask_batch[i,:slot_num+1] = 1

            # generate mask
            # Length of each captions + 2(start/end) is stored in nonzeros.
            # Eg : array([ 9, 11, 16, 12, 10])
            #nonzeros = np.array(list(map(lambda x: (x != 0).sum(), caption_batch[i])))
            #for ix, row in enumerate(mask_batch[i]):
            #    row[:nonzeros[ix]] = 1
            i+=1


        data = {}
        data['fc_feats'] = fc_batch
        data['face_feats'] = face_batch
        data['face_masks'] = face_mask_batch
        data['face_segment_ids'] = face_segment_batch
        data['captions'] = caption_batch
        data['masks'] = mask_batch
        data['position_ids'] = caption_position_batch
        data['segment_ids'] = caption_segment_batch
        data['blank_indexes'] =  blank_indexes_batch
        data['gt_captions'] = caption_gt_batch
        data["gt_masks"] = caption_gt_mask
        data["blank_masks"] = blank_masks_batch
        data['bert_emb'] = bert_batch
        data['characters'] = character_batch
        data['genders'] = gender_batch
        data['sent_num'] = sent_num_batch
        data['slots'] = slot_batch
        data['slot_masks'] = slot_mask_batch
        data['slot_size'] = np.max(slot_size)
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        return data

    # It's not coherent to make DataLoader a subclass of Dataset, but essentially, we only need to implement the following to functions,
    # so that the torch.utils.data.DataLoader can load the data according the index.
    # However, it's minimum change to switch to pytorch data loading.
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        return self.get_seg_batch(index), index

    def __len__(self):
        return len(self.info['videos'])

class SubsetSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

    # Add more in the queue
    def reset(self):
        """
        Two cases for this function to be triggered:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                            batch_size=1,
                                            sampler=SubsetSampler(self.dataloader.split_ix[self.split][self.dataloader.iterators[self.split]:]),
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=2, # 4 is usually enough
                                            collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped

    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()
        ix, wrapped = self._get_next_minibatch_inds()
        #tmp = self.split_loader.next()
        tmp = next(self.split_loader)

        if wrapped:
            self.reset()
        assert tmp[1] == ix, "ix not equal"
        return tmp + [wrapped]