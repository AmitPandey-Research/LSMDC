import json
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm 
import torch

path = "/scratch/amit/LSMDC16_info_fillin_new_augmented.json"

with open(path, 'r') as f:
    info_data = json.load(f)


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_tokens(['<blank>'], special_tokens=True)
tokenizer.add_tokens(['<blank_alert>'], special_tokens=True)

'''
for i in range(12):
    person_token = f"P{i+1}"
    tokenizer.add_tokens([person_token], special_tokens=True)
'''

tokenizer.add_tokens(['<sos>'], special_tokens=True)
tokenizer.add_tokens(['<eos>'], special_tokens=True)

print(f"Length of tokenizer is : {len(tokenizer)}")

tokenizer.save_pretrained("/scratch/amit/tokenizer/")


groups = info_data['groups']

# This is basically for giving Local IDs to the characters.
def get_character_batch(index):
    v_idx = groups[index]['videos']
    #print(f"V_idx is : {v_idx}")
    character_ids = []
    for id in v_idx:
        n_blanks = info_data['videos'][id]['num_blanks']
        #print(f"No.of blanks : {n_blanks}")
        if n_blanks > 0:
            characters = info_data['videos'][id]['character_id']
            #print(f"Characters: {characters}")
            for n in range(n_blanks):
                character_ids.append(characters[n])
    #print(f"Character IDs: {character_ids}")
    character_map = {}
    for c in character_ids:
        if c not in character_map:
            character_map[c] = len(character_map.keys()) + 1
    #print(f"Character map is : {character_map}")
    character_ids = [character_map[c] for c in character_ids]
    return character_ids


def _get_segment_position_ids(indexed_tokens): 
    sent_num = 0
    pos_count = 0
    segment_ids = []
    position_ids = []
    blank_indexes = []
    blank_masks = [] 
    for i, token in enumerate(indexed_tokens):
        if(token != 0):
            if(i != 0 and indexed_tokens[i-1] == 102):
                sent_num += 1
            if(indexed_tokens[i]  == 30522):
                blank_indexes.append(i)
            elif(indexed_tokens[i] == 30523):
                blank_masks.append(1)
            else:
                blank_masks.append(0)
            segment_ids.append(sent_num)
            position_ids.append(pos_count)
            pos_count += 1
        else:
            blank_masks.append(0)
            segment_ids.append(0)
            position_ids.append(0)

    return segment_ids, position_ids, blank_indexes, blank_masks


indexed_captions = []
gt_indexed_captions = []
tokenized_captions = []
gt_tokenized_captions = []
grouped_captions = []
ground_truth_captions = []
caption_length_tracker = []


for item in tqdm(info_data['groups']):
    index = item['id']
    v_idx = item['videos']
    split = item['split']
    captions = ['[CLS]']
    gt_captions = ['<sos>']
    if(split != "test"):
        character_ids = get_character_batch(index)
        char_id = 0
    for i, id in enumerate(v_idx):
        caption = info_data['videos'][id]['final_caption']
        caption = caption[1:-1] 
        caption.append('[SEP]')
        mod_caption = []
        gt_caption = []
        for i, x in enumerate(caption): 
            if x == "<UNK>":
                mod_caption.append('[UNK]')
                gt_caption.append('[UNK]')
            elif(x == "<blank>"):
                mod_caption.append('<blank_alert>')
                gt_caption.append('<blank_alert>')
                mod_caption.append("<blank>")
                if(split != "test"):
                    #person_id = f"P{character_ids[char_id]}"
                    person_id = f"{character_ids[char_id]}"
                    gt_caption.append(person_id)
                    char_id += 1
            else:
                mod_caption.append(x)
                gt_caption.append(x)
        captions.extend(mod_caption)  
        gt_captions.extend(gt_caption)
    
    
    captions = ' '.join(captions)
    #gt_captions.append("<eos>")
    gt_captions[-1] = "<eos>"
    gt_captions = ' '.join(gt_captions)

    grouped_captions.append(captions)
    ground_truth_captions.append(gt_captions)

    tokenized_text = tokenizer.tokenize(captions)
    gt_tokenized_text = tokenizer.tokenize(gt_captions)

    # Map the token strings to their vocabulary indeces.
    tokenized_captions.append(tokenized_text)
    gt_tokenized_captions.append(gt_tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    gt_indexed_tokens = tokenizer.convert_tokens_to_ids(gt_tokenized_text)
    indexed_captions.append(indexed_tokens)
    gt_indexed_captions.append(gt_indexed_tokens)

    segment_ids, position_ids, blank_indexes, blank_masks =_get_segment_position_ids(indexed_tokens)

    info_data['groups'][index]['group_caption'] = captions
    info_data['groups'][index]['index_caption'] = indexed_tokens
    info_data['groups'][index]['position_ids'] = position_ids
    info_data['groups'][index]['segment_ids'] = segment_ids
    info_data['groups'][index]['blank_indexes'] = blank_indexes
    info_data['groups'][index]['group_gt_caption'] = gt_captions
    info_data['groups'][index]['index_gt_caption'] = gt_indexed_tokens
    info_data['groups'][index]['blank_masks'] = blank_masks
    caption_length_tracker.append(len(indexed_tokens))

    #print(captions)
    #print(gt_captions)
    


tokenized_text = tokenized_captions[100]
indexed_tokens = indexed_captions[100]

# Display the words with their indeces.
for tup in zip(tokenized_text, indexed_tokens):
    print('{:<12} {:>6,}'.format(tup[0], tup[1]))


gt_tokenized_text = gt_tokenized_captions[100]
gt_indexed_tokens = gt_indexed_captions[100]

# Display the words with their indeces.
for tup in zip(gt_tokenized_text, gt_indexed_tokens):
    print('{:<12} {:>6,}'.format(tup[0], tup[1]))



print("Caption stats: ")
print(caption_length_tracker[:5])
print(np.max(caption_length_tracker), np.min(caption_length_tracker), np.mean(caption_length_tracker), np.median(caption_length_tracker), np.std(caption_length_tracker))
print(np.percentile(caption_length_tracker, [25, 50, 75, 90, 95, 99, 99.25, 99.50, 99.75, 99.99]))


save_path = "/scratch/amit/LSMDC16_info_fillin_sos_eos_blank_mask.json"

print(f"Saving info to path : {save_path}")

#with open(save_path, 'w') as f:
#    json.dump(info_data, f)

