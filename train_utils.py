import torch
import misc.utils as utils
import numpy as np
import time

def save_data(data, filename):
    base_save_path = "/scratch/amit/"
    np.save(base_save_path + filename, data)

def train_generator(gen_model, gen_optimizer, loader, is_overfit, grad_clip=0.1):
    gen_model.train()
    
    start_time = time.time()
    if(not is_overfit):
        data = loader.get_batch('train')
    else:
        data = loader.get_batch('overfit')
    end_time = time.time()
    
    print(f"Time taken to get train batch in train_utils.py : {end_time - start_time}")
    
    torch.cuda.synchronize()
    tmp = [data['fc_feats'], data['sent_num'], data['face_feats'], data['face_masks'], data['face_segment_ids'],
           data['captions'], data['masks'] , data['position_ids'], data['segment_ids'], data['blank_indexes'], data['gt_captions'], data["gt_masks"], data["blank_masks"], data['bert_emb'], data['slots'], data['slot_masks'], data['characters'], data['genders']]

    tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
    fc_feats, sent_num_batch, face_feats, face_masks, face_segment_ids, captions, masks, position_ids, segment_ids, blank_indexes, gt_captions, gt_masks, blank_masks, bert_emb, slots, slot_masks, characters, genders = tmp

    sent_num = data['sent_num']
    slot_size = data['slot_size']

    wrapped = data['bounds']['wrapped']
    gen_optimizer.zero_grad()


    #loss = gen_model(fc_feats, img_feats, face_feats, face_masks, captions, masks, bert_emb, slots, slot_masks, slot_size,
    #                 characters, genders)
    start_time = time.time()
    loss =  gen_model(fc_feats, sent_num_batch, face_feats, face_masks, face_segment_ids, captions, masks, position_ids, segment_ids, blank_indexes, gt_captions, gt_masks, blank_masks, bert_emb, slots, slot_masks, slot_size,
                characters, genders)
    end_time = time.time()
    print(f"Total time taken to run gen_model() - forward pass in train_utils.py : {end_time - start_time}")
    
    loss = loss.mean()
    start_time = time.time()
    loss.backward()
    end_time = time.time()
    print(f"Total time taken to run gen_model() - backward() pass in train_utils.py : {end_time - start_time}")
    gen_loss = loss.item()

    utils.clip_gradient(gen_optimizer, grad_clip)
    gen_optimizer.step()
    torch.cuda.synchronize()

    return gen_loss, wrapped, sent_num