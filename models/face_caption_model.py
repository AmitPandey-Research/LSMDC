import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.sent_embedding import SentEmbedding
from .ftv_encoder import FTV_encoder
from transformers import BertConfig
from transformers import BertModel
from transformers import BertTokenizer
import numpy as np
import time

#import ipdb
import time

class FaceCaptionModel(nn.Module):
    def __init__(self, opt):
        super(FaceCaptionModel, self).__init__()
        self.memory_encoding_size = opt.encoding_size
        self.face_encoding_size = opt.face_encoding_size
        print(f"Mem encoding size is : {self.memory_encoding_size}")
        self.batch_size = opt.batch_size
        self.classify_gender = opt.classify_gender
        self.max_caption_length = 120

        self.use_text_encoder = True

        # Bert Modelling
        if(self.use_text_encoder):
            tokenizer_save_path = "/scratch/amit/tokenizer/"
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_save_path)
            #configuration = BertConfig(type_vocab_size=5, output_hidden_states = True)
            self.TOKEN_LENGTH = len(self.tokenizer)  #  With <blank> and P1-P11 added,<sos>,<eos>
            print(f"Vocab size is : {self.TOKEN_LENGTH}")
            self.CLS_TOKEN_ID = 101
            self.SEP_TOKEN_ID = 102
            self.START_TOKEN = 30536
            #self.END_TOKEN = 30537
            self.END_TOKEN = 13
            self.BLANK_ALERT = 30523
            self.MAX_CAPTIONS = 5
            #ipdb.set_trace()
            #self.text_encoder_embedding_layer = self.bert_model.get_input_embeddings()
            #self.segment_embed = self.bert_model.embeddings.token_type_embeddings

        # Caption Embedding
        self.caption_embedding = nn.Embedding(self.TOKEN_LENGTH, self.memory_encoding_size)

        # BERT Encode
        self.bert_encode = nn.Linear(opt.bert_size, self.memory_encoding_size)

        # Caption Position + Segment Embeddings
        self.position_embed = nn.Embedding(self.max_caption_length, self.memory_encoding_size)
        self.segment_embed = nn.Embedding(5, self.memory_encoding_size)

        # Feature Conversion
        self.video_encode = nn.Linear(1024, self.memory_encoding_size).cuda()
        self.face_encode = nn.Linear(512, self.memory_encoding_size).cuda()

        # Character Decoder
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.memory_encoding_size, nhead=8, batch_first = True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)

        num_classes = opt.unique_characters + 1

        # Encoder 
        self.encoder = FTV_encoder(opt)
        self.dense = nn.Linear(self.memory_encoding_size, num_classes)


        self.person_classifier = nn.Linear(self.memory_encoding_size,12)
        self.word_classifier = nn.Linear(self.memory_encoding_size,30526)
        self.word_classifier2 = nn.Linear(self.memory_encoding_size,self.TOKEN_LENGTH)
        ## 30537 is eos , 30536 is sos, 30524 to 30535 P1-P12. 30522 blank
        ## these in nn caption embedding. that we trying to split using
        ## these two classifiers.

        



        # Gender Classification 
        if(self.classify_gender):
            self.gender_face_embed = nn.Linear(self.face_encoding_size, self.face_encoding_size)
            self.gender_logit = nn.Sequential(
            nn.Linear(self.memory_encoding_size, self.memory_encoding_size),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(self.memory_encoding_size, 3),
            )
            
            self.gender_loss_weight = opt.gender_loss


        self.logits = nn.Linear(self.memory_encoding_size, self.TOKEN_LENGTH)
    
        # Loss function
        self.softmax = nn.Softmax(dim=1)
        self.cross_loss = nn.CrossEntropyLoss(ignore_index = -100)

        # Person Loss function
        self.person_loss = PersonLoss(num_classes)


        self.token_to_person_id_map = self._populate_map()
    

    def _populate_map(self):
        token_to_person_id_map = {}
        start_person_id_token = 30524

        for i in range(1, 12):
            token_to_person_id_map[start_person_id_token] = i
            start_person_id_token += 1
        
        token_to_person_id_map[0] = 0

        return token_to_person_id_map


  
    def _get_segment_position_ids(self, indexed_tokens):
        final_segment_ids = []
        final_position_ids = []
        final_sentence_to_blank_indexes = []
        final_blank_indexes = torch.zeros((self.batch_size, 17), dtype=int)
        for bs, index_token in enumerate(indexed_tokens):
            sent_num = 0
            pos_count = 0
            segment_ids = []
            position_ids = []
            blank_indexes = []
            sent_to_blank_index = {}
            for i, token in enumerate(index_token):
                if(token != 0):
                    if(i != 0 and index_token[i-1] == 102):
                        sent_num += 1
                    if(index_token[i]  == 30522):
                        blank_indexes.append(i)
                        if(sent_num not in sent_to_blank_index):
                            sent_to_blank_index[sent_num] = []
                            sent_to_blank_index[sent_num].append(i)
                    segment_ids.append(sent_num)
                    position_ids.append(pos_count)
                    pos_count += 1
                else:
                    segment_ids.append(0)
                    position_ids.append(0)
            
            final_segment_ids.append(segment_ids)
            final_position_ids.append(position_ids)
            final_sentence_to_blank_indexes.append(sent_to_blank_index)
            final_blank_indexes[bs,:len(blank_indexes)] = torch.tensor(blank_indexes)

        tokens_tensor = torch.tensor(indexed_tokens)
        segments_tensors = torch.tensor(final_segment_ids).cuda()
        position_tensors = torch.tensor(final_position_ids).cuda()
        final_blank_indexes = final_blank_indexes.cuda()

        return  tokens_tensor, segments_tensors, position_tensors, final_sentence_to_blank_indexes, final_blank_indexes

    def _get_cls_blank_combined(self, tokens_tensor, last_hidden_state, sentence_to_blank_indexes):

        all_captions = []
        #print(tokens_tensor.shape)
        for i, token_tensor in enumerate(tokens_tensor):
            #print(token_tensor, token_tensor.shape)
            CLS_indexes = token_tensor == self.CLS_TOKEN_ID
            cls_embedding =  last_hidden_state[i][CLS_indexes]
            sent_embeddings = []
            num_sentences = (token_tensor == self.SEP_TOKEN_ID).sum()
            s_embedings = torch.zeros((self.MAX_CAPTIONS, 768))
            #print(f"No.of sentences : {num_sentences}")
            for index in range(0, num_sentences):
                blank_positions = sentence_to_blank_indexes[i].get(index, [])
                for b_i, pos in enumerate(blank_positions):
                    if(b_i == 0):
                        s_emb = cls_embedding + last_hidden_state[i][pos]
                    s_emb += last_hidden_state[i][pos]
                    sent_embeddings.append(s_emb)
                if(len(blank_positions) == 0):
                    sent_embeddings.append(cls_embedding)

            sentence_tensor = torch.stack(sent_embeddings).squeeze(1)

            s_embedings[:num_sentences, :] = sentence_tensor
            all_captions.append(s_embedings)

        
        return torch.stack(all_captions)
    
    def get_causal_mask(self, seq_length):
        # Create a square matrix of ones with shape (seq_length, seq_length)
        causal_mask = torch.ones(seq_length, seq_length)

        # Set the upper triangular elements to a large negative value (-inf)
        causal_mask = torch.triu(causal_mask, diagonal=1)

        # Repeat the mask for each element in the batch
        #causal_mask = causal_mask.unsqueeze(0).repeat(self.batch_size, 1, 1)

        return causal_mask.bool().cuda()

    def _get_person_ids(self, predictions):
        final_pred = torch.zeros_like(predictions)

        for i, pred in enumerate(predictions):
            final_pred[i] =  self.token_to_person_id_map.get(pred.item(), 0)
        
        return final_pred
    
    def _extract_blank_outputs(self, dec_output, blank_indexes):
        blank_outputs = []
        for i, blank_index in enumerate(blank_indexes):
            blank_index = blank_index[blank_index.nonzero()]
            if(len(blank_index) != 0):
                blank_output = dec_output[i, blank_index, :]
                blank_output = blank_output.squeeze(1)
                blank_outputs.append(blank_output)
        
        blank_outputs = torch.cat(blank_outputs)
        return blank_outputs
    
    def _flattened_person_ids(self, characters):
        person_ids = []
        for i, character in enumerate(characters):
            character = character[character.nonzero()]
            if(len(character) != 0):
                person_ids.extend(character)

        return torch.cat(person_ids)

    def _flatten_genders(self, genders, slots):
        gender_ids = []
        for i, gender in enumerate(genders):
            gender = gender[slots[i].nonzero()]
            if(len(gender) != 0):
                gender_ids.extend(gender)
        
        return torch.cat(gender_ids)
    
    def autoregressive_decoder(self, encoder_output, gt_captions):

        predictions = gt_captions[:, 0].reshape(self.batch_size,1)
        # 16 x 1
        #predictions = np.repeat(start_token_index, self.batch_size).reshape(self.batch_size,1)
        print("predictions with start token", predictions.shape, predictions)

        # Encoder_output : 16 x 92 x 512
        
        # gt_caption_decoder : 16 x 120 x 512 

        # 16 x 1 x 512 --> (start_token)
        for i in range(self.max_caption_length):
            
            predictions_emb = self.caption_embedding(torch.tensor(predictions))
            dec_output = self.transformer_decoder(predictions_emb, encoder_output)
            logits = self.logits(dec_output)
            _, tgt = torch.max(logits, 2)
            tgt = tgt[:,-1].reshape(self.batch_size, 1)
            predictions = np.hstack((predictions, tgt))


        #print("shape of predictions in fill in auto reg is: ", predictions.shape)

        return predictions

    
    def fill_in_autoregressive_decoder(self, encoder_output, gt_captions, caption_pos_ids , caption_seg_ids, encoder_masks):
        predictions = torch.zeros((self.batch_size, self.max_caption_length), dtype=int).cuda()
        predictions[:,0] = gt_captions[:,0]

        for i, gt_caption in enumerate(gt_captions):
            single_encoder_output = encoder_output[i] # 92 x 512
            encoder_mask = encoder_masks[i]
            for index, token in enumerate(gt_caption):
                #print(index, predictions[i, :index+1])
                if(token.item() == self.END_TOKEN):
                    break
                if(index == 0):
                    tgt = gt_caption[index + 1]
                    predictions[i, index + 1] = tgt
                    continue
                if(gt_caption[index].item() == self.BLANK_ALERT):
                    #ipdb.set_trace()
                    predictions_emb = self.caption_embedding(predictions[i, :index+1]) # torch.Size([4, 512])
                    # Adding position IDs and segment IDs
                    predictions_emb = predictions_emb + caption_pos_ids[i, :index+1] + caption_seg_ids[i, :index+1]
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(predictions_emb.shape[0]).cuda()
                    dec_output = self.transformer_decoder(predictions_emb, single_encoder_output, 
                                                          tgt_mask=tgt_mask, 
                                                          memory_key_padding_mask=encoder_mask)
                    logits = self.dense(dec_output)
                    _, tgt = torch.max(logits, 1)
                    next_token = tgt[-1]
                    predictions[i, index + 1] = next_token
                else:
                    tgt = gt_caption[index + 1]
                    predictions[i, index + 1] = tgt

        print("shape of predictions in fill in auto reg is: ", predictions.shape)
        
        
        return predictions

    

    def fillin_autoregressive_decoder(self, batch_memory, batch_input_embedding, batch_Blank_indices, batch_num_BlankAlert = None,
    batch_target = None, batch_size = 16, memory_mask = None, memory_key_padding_masks = None, use_causal_mask = True):
  
        '''
        We will pass a batch as input for prediction. The batch will consist of the following:
        1.batch_memory == encoder output --> BS x SL x DIM
        2.batch_input_embedding --> embedding having blanks --> BS x SL x Dim ==> i.e. having emb = token emb + segment emb + position emb for each token
        
        ex: [[<sos> <Blank_Alert> <Blank> hit <Blank_Alert> <Blank> <sep>.....<eos>],....]
        
        
        3.batch_BlankAlert_indices --> a list of list contatining the indices of <Blank_Alert> tokens in the given caption. ex: [[1,4],...]
        4.batch_num_BlankAlert --> a list having total number of blanks (equal to total number of Blank Alerts) for each sequence in the batch
        5.batch_target --> a list of list contatining ground truth corresponding to real caption
        6.batch_size
        

        '''
        loss = 0
        pred = []

        for i in range(batch_size):
            n = len(batch_Blank_indices[i])
            memory_key_padding_mask = memory_key_padding_masks[i].cuda()

            print("blank indexes of current seq: ", batch_Blank_indices[i] )

            Blank_indices_of_seq = batch_Blank_indices[i] ##has zeros as pad
            Blank_indices_of_seq = Blank_indices_of_seq[Blank_indices_of_seq>0]




            for b_index in Blank_indices_of_seq:
                all_tokens_emb_upto_ba_index = batch_input_embedding[i][:b_index].cuda()

                print("b_index: ", b_index)

                tgt_mask = torch.torch.ones(b_index,b_index).cuda()

                if use_causal_mask == True:
                    tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(b_index).cuda()



                dec_output = self.transformer_decoder(all_tokens_emb_upto_ba_index,batch_memory[i], tgt_mask = tgt_mask, memory_key_padding_mask = memory_key_padding_mask, memory_mask = memory_mask)
                logits = self.dense(dec_output)
                print("logits shape :", logits.shape)
                #print(" logits are :", logits)
                _ , tgt = torch.max(logits,1)
                print("tgt shape: ", tgt.shape)


                Person_ID = tgt[-1]

                pred = pred + [Person_ID] # One list for whole batch.

                batch_input_embedding[i][b_index] = self.caption_embedding(torch.tensor(Person_ID))

        return torch.tensor(pred)

    def caption_autoregressive_decoder(self, batch_memory, batch_seg_ids, batch_position_ids, batch_target = None,   batch_target_pad_mask = None, 
    BA_token_index = None, sep_token_index = None, start_token_index = None, batch_input_embedding = None, batch_Blank_indices = None, batch_num_BlankAlert = None, 
                                   batch_size = 2, memory_mask = None,memory_key_padding_masks = None, mode = 'train'):

        print(" function called")
        function_call_time = time.time()


        batch_memory = batch_memory.transpose(0,1)
        EOS_token = 30536
        P1_token = 30524
        P12_token = 30535
        
        print("segment ids: ",batch_seg_ids[0])
        print("caption: ", batch_target[0])
  
        '''
        BLANK_ALERT = 30523
        3:12
        BLANK is 30522
        3:14
        START_TOKEN = 30536 and END_TOKEN = 30537 but we subtract to 30524 to get the corresponding correct Person ID tokens.
        So it becomes <sos> -- 12 and <eos>--13

        <blank>": 30522,': 1,
    '  "<blank_alert>": 30523,': 2,
    '  "<eos>": 30537,': 3,
    '  "<sos>": 30536,': 4,
    '  "P1": 30524,': 5,
    '  "P10": 30533,': 6,
    '  "P11": 30534,': 7,
    '  "P12": 30535,': 8,
    '  "P2": 30525,': 9,
    '  "P3": 30526,': 10,
    '  "P4": 30527,': 11,
    '  "P5": 30528,': 12,
    '  "P6": 30529,': 13,
    '  "P7": 30530,': 14,
    '  "P8": 30531,': 15,
    '  "P9": 30532



        '''

        '''
        I am using 0 t0 11 as person id

        currently teacher forcing the whole caption but the Person IDS. Those we are predicting.

         ~ gt_captions_mask.bool().cuda() --- caption pad mask

        '''



        if mode == 'train':
            #print(" batch target pad mask:", batch_target_pad_mask.shape)
            loss = 0
            batch_pred_caption = []

            for i in range(batch_size):

                print("##### Batch Index ####: ", i)
                print(' batch size: ', batch_size)
                print('self.batch_size: ',self.batch_size)
                
                
                print('self.TOKEN_LENGTH: ', self.TOKEN_LENGTH)

                seq_pred_caption = []
                
                segment_id_list = [0]
                position_id_list = [0]

                print(" memory key padding maskS shape: ", memory_key_padding_masks.shape)

                
                memory_key_padding_mask = memory_key_padding_masks[i]#.cuda()

                #print(" memory key padding mask shape: ", memory_key_padding_mask.shape)

                #print(' memory size', batch_memory[i].shape)

                caption_padding_mask = batch_target_pad_mask[i].cuda()

                #print('caption_padding_mask shape: ', caption_padding_mask.shape)

                caption_segment_id = batch_seg_ids[i]

                caption_position_id = batch_position_ids[i]
                

                target = batch_target[i]

                targets_for_loss = target.clone()
                input_to_decoder = target.clone()
                EOS_idx = torch.argwhere(input_to_decoder == 30537)

                print("EOS IDX :", EOS_idx)

                logits_for_loss = torch.zeros((EOS_idx,self.TOKEN_LENGTH)).cuda() ## not 119, self.token_length



                # <sos> <BA> P3 slaps <BA> P3.
                # <sos> <BA> P1 slaps <BA> P1.


                ## find max length till eos
                ## teacher force person id


                # total 30538

                #  0 t0 30523, 30524 to 30535, 30536,30537


                #  102,30522,30537

                #  30538

                #  loss 


            # for batch ---- max eos for whole sequence...

            # logits we have predicted

            # mask based on person

            

                for j in range(EOS_idx):

                
                    caption_padding_mask_till_j = caption_padding_mask[:j+1]
                    print("caption pad mask shape: ", j , caption_padding_mask_till_j.shape)

                    caption_segment_id_till_j = caption_segment_id[:j+1]

                    caption_position_id_till_j = caption_position_id[:j+1]

                    print("memory key padding mask shape: ", memory_key_padding_mask.shape)





                    tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(j+1).cuda()

                    tgt = self.caption_embedding(torch.tensor(input_to_decoder[:j+1])) + self.segment_embed(torch.tensor(caption_segment_id_till_j)) + self.position_embed(torch.tensor(caption_position_id_till_j))
 
                    print("tgt shape: ", tgt.shape)
                    print(" memory_key_padding_mask: ",memory_key_padding_mask.shape)
                    print('caption_padding_mask_till_j', caption_padding_mask_till_j.shape)
                    print('batch_memory[i] shape: ', batch_memory[i].shape)

                    dec_output = self.transformer_decoder(tgt,batch_memory[i], tgt_mask = tgt_mask, memory_key_padding_mask = memory_key_padding_mask, memory_mask = memory_mask, tgt_key_padding_mask = caption_padding_mask_till_j)

                    if input_to_decoder[j] == BA_token_index:
                        logits = self.person_classifier(dec_output) ## linear of output 12
                        print(" logits shape: ", logits.shape)
                        _ , tgt = torch.max(logits,1)
                        print("tgt shape: ",tgt.shape)
                        Person_ID = tgt[-1] + P1_token ## Since P1 starts at 30524
                        logits_for_loss[j,P1_token:P12_token+1] = logits[-1]
                        print('logits for loss indexed inside person: ',logits_for_loss[j].shape, logits_for_loss[j,30523:30537])
                        seq_pred_caption.append(Person_ID.item())
                        #target[j+1] = Person_ID.item()


                    else:
                        logits = self.word_classifier(dec_output) ## Size = 30526 = 30522 + special tokens (not person ids) [ Total vocab -  person ids i.e. 12 ==  size of vocab]
                        _ , tgt = torch.max(logits,1)
                        #print('logits value  ',logits[-1], logits[-1].shape)
                        logits_for_loss[j,:P1_token] = logits[-1,:P1_token]
                         
                        logits_for_loss[j,P12_token+1: self.TOKEN_LENGTH] = logits[-1,30524:30526] ## for word classifier, total size is 30526, so we need 30524 and 30525

                        print('logits for los []indexed inside word: ',logits_for_loss[j],logits_for_loss[j].shape,max(logits_for_loss[j]))

                        


                        Word_ID = tgt[-1]
                        if Word_ID == 30525: ## eos in word classifier
                            Word_ID = torch.tensor(30537) ##eos position in caption embedding
                        print("word id: ", Word_ID, type(Word_ID))
                        seq_pred_caption.append(Word_ID.item())


                seq_pred_caption = torch.tensor(seq_pred_caption)

                print("seq pred caption size: ", len(seq_pred_caption))

                cmask = caption_padding_mask[1:]

                print("cmask: ", cmask)

                #print('seq_pred_caption padding: ', seq_pred_caption, seq_pred_caption.shape)
                #print('caption padding mask: ', caption_padding_mask)

               

                #print('cmask shape: ', cmask.shape)
                #print('targets for loss shape: ', targets_for_loss.shape)

                targets_for_loss[caption_padding_mask] = -100
                targets_for_loss = targets_for_loss[:EOS_idx+1] ## now predicting only till eos
                print('seq_pred_caption padding: ', seq_pred_caption, seq_pred_caption.shape)
                #loss = 0
                print(" j before loss: ",j)
                loss = loss + self.cross_loss(logits_for_loss, targets_for_loss[1:])
                print("loss", loss)

                batch_pred_caption.append(seq_pred_caption)
            print("loss calculated")
            loss_calc_time = time.time()
            print("time from func call to loss: ", loss_calc_time - function_call_time)
            loss = loss/batch_size
            return loss




        # if mode == 'predict':
        #     loss = 0
        #     batch_pred_caption = []

        #     for i in range(batch_size):

        #         seq_pred_caption = []
        #         tgt_seq = [start_token_index]
        #         segment_id_list = [0]
        #         position_id_list = [0]

                
        #         memory_key_padding_mask = memory_key_padding_masks[i].cuda()

        #         #caption_padding_mask = batch_target_pad_mask[i].cuda()

        #         #caption_segment_id = batch_seg_ids[i]

        #         #caption_position_id = batch_position_ids[i]

        #         target = batch_target[i]
            



            

        #         for j in range(119):

                    

                
        #             #caption_padding_mask_till_j = caption_padding_mask[:j+1]

        #             caption_segment_id_till_j = segment_id_list[:j+1]

        #             caption_position_id_till_j = position_id_list[:j+1]

        #             # segment_id = 0

        #             # sos t1 t2 sep  t3 t4  eos
        #             # 0   0   0   1  1    



        #             tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(j+1).cuda()

        #             tgt = self.caption_embedding(torch.tensor(tgt_seq[:j+1])) + self.segment_embed(torch.tensor(caption_segment_id_till_j)) + self.position_embed(torch.tensor(caption_position_id_till_j))

        #             dec_output = self.transformer_decoder(tgt,batch_memory[i], tgt_mask = tgt_mask, memory_key_padding_mask = memory_key_padding_mask, 
        #             memory_mask = memory_mask)

        #             if tgt_seq[j] == BA_token_index:
        #                 logits = self.person_classifier(dec_output) ## linear of output 12
        #                 _ , tgt = torch.max(logits,1)
        #                 Person_ID = tgt[-1] + 30524
        #                 seq_pred_caption.append(Person_ID.item())
        #                 tgt_seq.append(Person_ID)


        #             else:
        #                 logits = self.word_classifier(dec_output) ## 30052 + special tokens (not person ids)
        #                 _ , tgt = torch.max(logits,1)
        #                 Word_ID = tgt[-1]

        #                 if Word_ID == 30525: ## eos in word classifier
        #                     Word_ID == 30537 ##eos position in caption embedding

        #                 if Word_ID == 30524: ## sos in word classifier
        #                     Word_ID = torch.tensor(30536) ## sos position in caption embedding    
        #                 seq_pred_caption.append(Word_ID.item())
        #                 tgt_seq.append(Word_ID)

        #                 if Word_ID == sep_token_index:
        #                     segment_id = segment_id+1 




        #                 position_id_list.append(j+1)
        #                 segment_id_list.append(segment_id)






        #         batch_pred_caption.append(seq_pred_caption)


                


        #     return batch_pred_caption   

    

    def caption_autoregressive_decoder2(self, batch_memory, batch_seg_ids, batch_position_ids, batch_target = None,   batch_target_pad_mask = None, 
    BA_token_index = None, sep_token_index = None, start_token_index = None, batch_input_embedding = None, batch_Blank_indices = None, batch_num_BlankAlert = None, 
                                   batch_size = 2, memory_mask = None,memory_key_padding_masks = None, target_type = None, len_mask = None, seq_lens = None, mode = 'train'):

        print(" function called")
        function_call_time = time.time()

        if mode == 'train':

            print('self.TOKEN_LENGTH: ', self.TOKEN_LENGTH)

            target = batch_target

            targets_for_loss = target.clone()

            print("targets_for_loss shape: ",targets_for_loss.shape)


            input_to_decoder = batch_input_embedding
            

            batch_memory = batch_memory.transpose(0,1)
            
            

            print(' memory size', batch_memory.shape)


            print(" memory key padding maskS shape: ", memory_key_padding_masks.shape)


            
            batch_pred_caption = []

            max_batch_seq_len = max(seq_lens).to(dtype=torch.long)

            print("max_batch_seq_len: ", max_batch_seq_len)

            print("batch_target_pad_mask shape: ",batch_target_pad_mask)
                
            loss = 0
            for t in range(max_batch_seq_len):        

                        print("t: ",t) 


                        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(t+1).cuda()
                        print("tgt mask shape:", tgt_mask.shape)

                        tgt = input_to_decoder[:,:t+1]
                        tgt_key_padding_mask = batch_target_pad_mask[:,:t+1]
                        print("tgt key padding mask shape: ", tgt_key_padding_mask.shape)

                        print("tgt shape: ", tgt.shape)
                        print(" memory_key_padding_mask: ",memory_key_padding_masks.shape)
                        

                        dec_output = self.transformer_decoder(tgt,batch_memory, tgt_mask = tgt_mask, memory_key_padding_mask = memory_key_padding_masks, memory_mask = memory_mask, tgt_key_padding_mask = tgt_key_padding_mask)
                        print("dec_output shape: ", dec_output.shape)


                        feat = dec_output[:,t,:]

                        print("feat shape: ", feat.shape)

                        valid_feat = feat[len_mask[:,t].bool()]

                        print("valid_feat shape: ", valid_feat.shape)
                        


                        word_logits = self.word_classifier2(valid_feat)  # 2 x 30000
                        peid_logits = self.person_classifier(valid_feat)  # 2 x 12

                        print("word logits and peid logits shape: ", word_logits.shape,peid_logits.shape)
                        print("peid logits: ", peid_logits)
                        # the list of samples where at timestamp t, you have words or personids
                        # batch_data['target_type'][:, t] for t == 0 is [2, 1]
                        word_idx = target_type[:, t] == 1  # [False, True]
                        peid_idx = target_type[:, t] == 2  # [True, False]

                        print("word_idx and peid_idx:", word_idx, peid_idx)

                        print("targets_for_loss[:,t] shape: ",targets_for_loss[:,t].shape)
                        print("targets for loss: ", targets_for_loss[:,t])
                        

                        if torch.any(word_idx):
                            print(" calculating loss for word_idx")
                            loss += self.cross_loss(word_logits[word_idx], targets_for_loss[:,t][word_idx])
                            print("word loss calculated successfully")
                        if torch.any(peid_idx):
                            loss += self.cross_loss(peid_logits[peid_idx], targets_for_loss[:,t][peid_idx])

        

                        print("loss", loss)
            print("batch loss: ", loss)
            print("loss calculated")
            loss_calc_time = time.time()
            print("time from func call to loss: ", loss_calc_time - function_call_time)
            
        return loss

    





    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_' + mode)(*args, **kwargs)

    def _forward(self, fc_feats, sent_num_batch, face_feats, face_masks, face_segment_ids, captions, caption_masks, position_ids, segment_ids, blank_indexes, gt_captions, gt_captions_mask, blank_masks, bert_emb, slots, slot_masks, slot_size,
                characters, genders=None):

        # print("gt caption mask: ", gt_captions_mask)
        # print(" blank mask: ", blank_masks)
        # print("blank indexes: ", blank_indexes)
        # print("captions: ", captions)
        # return 


        


        #ipdb.set_trace()
        # bs x 17 x 512

        # set the last column to zero if a blank is present
        # Last position ignore the blank since we cut off at length 120.
        blank_masks[:, -1].fill_(0)

        gt_mask = blank_masks.bool()
        column_of_zeros = torch.zeros(blank_masks.shape[0], 1).cuda()
        pred_mask = torch.cat((blank_masks[:, 1:], column_of_zeros), dim=1)

        target_type = torch.zeros_like(blank_masks)
        target_type[blank_masks.bool()] = 1
        target_type[~blank_masks.bool()] = 2
        target_type = target_type * gt_captions_mask.bool()
        seq_lens = gt_captions_mask.sum(axis=1)

        text_embedding = self.bert_encode(bert_emb) # 17 x 1536 --> 17 x 512

        
        # bs x 50 x 512 --  bs x 5 x 10 x 512
        face_features = face_feats.reshape(self.batch_size, face_feats.shape[1] * face_feats.shape[2], face_feats.shape[3])
        if(face_features.shape[-1] != self.memory_encoding_size):
            face_features = self.face_encode(face_features)
        
    
        video_embedding = self.video_encode(fc_feats)
        video_embedding = video_embedding.reshape(self.batch_size, video_embedding.shape[1] * video_embedding.shape[2], video_embedding.shape[3])

        start_time = time.time()
        # bs x 92 x 512 [ 25 + 50 + 17]
        encoder_output, input_masks = self.encoder(text_embedding, video_embedding, sent_num_batch, face_features, face_masks, face_segment_ids, slots, slot_masks, self.segment_embed)
        print(" encoder output shape", encoder_output.shape)
        end_time = time.time()
        #print(f"Time taken for self.encoder() in _forward: {end_time - start_time}")
    
        start_time = time.time()

        gt_captions = torch.where(gt_captions >= 30524, gt_captions - 30524, gt_captions)
        # bs x 120
        gt_caption_embedding = self.caption_embedding(gt_captions)
        caption_pos_ids = self.position_embed(position_ids)
        # To-DO : Share the segments IDs/ video IDS with encoder.
        caption_seg_ids = self.segment_embed(segment_ids)

        gt_caption_embedding = gt_caption_embedding + caption_pos_ids + caption_seg_ids



        loss = self.caption_autoregressive_decoder2(batch_memory = encoder_output, batch_target = gt_captions, batch_seg_ids = segment_ids,
        batch_position_ids = position_ids, 
        batch_target_pad_mask = ~ gt_captions_mask.bool().cuda(), BA_token_index = 30523, sep_token_index = 102, start_token_index = 30536, batch_input_embedding = gt_caption_embedding, 
        batch_Blank_indices = None, batch_num_BlankAlert = None, 
                                   batch_size = 2, memory_mask = None,memory_key_padding_masks = input_masks, target_type = target_type, len_mask = gt_captions_mask, seq_lens = seq_lens, mode = 'train')

        return loss

        '''
        Naveen original
        
        #ipdb.set_trace()
        gt_captions = torch.where(gt_captions >= 30524, gt_captions - 30524, gt_captions)

        # bs x 120
        gt_caption_embedding = self.caption_embedding(gt_captions)
        caption_pos_ids = self.position_embed(position_ids)
        # To-DO : Share the segments IDs/ video IDS with encoder.
        caption_seg_ids = self.segment_embed(segment_ids)

        gt_caption_embedding = gt_caption_embedding + caption_pos_ids + caption_seg_ids
        end_time = time.time()

        #ipdb.set_trace()
        # bs x 120 
        tgt_key_padding_mask = ~ gt_captions_mask.bool().cuda()
        #ipdb.set_trace()
        #tgt_mask = self.get_causal_mask(gt_caption_embedding.shape[1])
        # causal mask : seq_len x seq_len - 120x120
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(gt_caption_embedding.shape[1]).cuda()
        start_time = time.time()
        dec_output = self.transformer_decoder(gt_caption_embedding.transpose(0,1), encoder_output, tgt_mask = tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                                                                             memory_key_padding_mask = input_masks).transpose(0,1)
        end_time = time.time()

        #ipdb.set_trace()
        # bs x 120 ( 0,0,1,0,0,0)
        # no.of blanks x 512
        logits = self.dense(dec_output[pred_mask.bool()])
        # Define the target labels for the masked tokens,
        # Remove the <sos> part of the gt_caption.
        labels = gt_captions.clone()
        labels = labels[gt_mask]
        
        loss = self.cross_loss(logits, labels) '''

        #return loss

    def _predict(self, fc_feats, sent_num_batch, face_feats, face_masks, face_segment_ids, captions, caption_masks, position_ids, segment_ids, blank_indexes, gt_captions, gt_captions_mask, blank_masks, bert_emb, slots, slot_masks, 
                 slot_size):
        
    
        batch_size = fc_feats.size(0)
        masks = slot_masks[:, :slot_size + 1].bool()


        #tokens_tensor, segments_tensors, position_tensors, sentence_to_blank_indexes, blank_indexes = self._get_segment_position_ids(captions)
        #ipdb.set_trace()

        text_embedding = self.bert_encode(bert_emb) # 17 x 1536 --> 17 x 512

        #gt_caption_embedding = self.text_encoder_embedding_layer(captions)
        gt_captions = torch.where(gt_captions >= 30524, gt_captions - 30524, gt_captions)

        gt_caption_embedding = self.caption_embedding(gt_captions)
        caption_pos_ids = self.position_embed(position_ids)
        caption_seg_ids = self.segment_embed(segment_ids)

        gt_caption_embedding = gt_caption_embedding + caption_pos_ids + caption_seg_ids
        

        #ipdb.set_trace()
        # bs x 17 x 768
        '''
        cls_blank_combined = self._get_cls_blank_combined(tokens_tensor, last_hidden_state, sentence_to_blank_indexes)
        cls_blank_combined = cls_blank_combined.cuda()
        text_embedding = self.caption_encode(cls_blank_combined)
        '''
        # bs x 50 x 512 --  bs x 5 x 10 x 512
        face_features = face_feats.reshape(self.batch_size, face_feats.shape[1] * face_feats.shape[2], face_feats.shape[3])
        if(face_features.shape[-1] != self.memory_encoding_size):
            face_features = self.face_encode(face_features)
        video_embedding = self.video_encode(fc_feats)
        video_embedding = video_embedding.reshape(self.batch_size, video_embedding.shape[1] * video_embedding.shape[2], video_embedding.shape[3])

        #text_embedding = None
        #ipdb.set_trace()
        encoder_output, input_masks  = self.encoder(text_embedding, video_embedding, sent_num_batch, face_features, face_masks, face_segment_ids, slots, slot_masks, self.segment_embed)


        #predictions = self.autoregressive_decoder(encoder_output, gt_captions)
        #ipdb.set_trace()
        predicted_caption = self.fill_in_autoregressive_decoder(encoder_output.transpose(0,1), gt_captions, caption_pos_ids , caption_seg_ids, input_masks)
        print(" shape returned to  _predict ", predicted_caption.shape)
        predicted_person_tokens = torch.masked_select(predicted_caption, blank_masks.bool())

        print(" shape after blank mask token selection ", predicted_person_tokens.shape)
        #predictions = self._get_person_ids(predicted_person_tokens)
        predictions = predicted_person_tokens
        print(f"Predictions : {predictions}")



        # predicted_from_sliced_autoreg = self.fillin_autoregressive_decoder(batch_memory = encoder_output.transpose(0,1), batch_input_embedding = gt_caption_embedding, batch_Blank_indices = blank_indexes, memory_key_padding_masks = input_masks )
        # print(" shape of predicted from sliced autoreg: ", predicted_from_sliced_autoreg.shape)
        # print( " predicted from sliced autoreg", predicted_from_sliced_autoreg )


        '''
        tgt_key_padding_mask = ~ gt_captions_mask.bool().cuda()
        #ipdb.set_trace()
        tgt_mask = self.get_causal_mask(gt_caption_embedding.shape[1])
        dec_output = self.transformer_decoder(gt_caption_embedding.transpose(0,1), encoder_output, tgt_mask = tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                                                                             memory_key_padding_mask = input_masks).transpose(0,1)
        
        logits = self.logits(dec_output)
        _, tgt = torch.max(logits, 2)
        predictions = torch.masked_select(tgt, blank_masks.bool())
        predictions = self._get_person_ids(predictions)
        '''
       
        if(self.classify_gender):
            gender_logits = self.gender_logit((self.gender_face_embed(output)))
            _, predicted_genders = torch.max(gender_logits, 1)
        else:
            predicted_genders = predictions.new_zeros(predictions.size(0),dtype=torch.long)
        
        return predictions, predicted_genders
    


class PersonLoss(nn.Module):
    def __init__(self, vocab_size):
        super(PersonLoss, self).__init__()
        self.vocab_size = vocab_size
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, logits, labels):
        # logits: [batch_size, seq_len, vocab_size]
        # labels: [batch_size, seq_len]
        # mask: [batch_size, seq_len]
        
        #ipdb.set_trace()
        # Flatten the logits and labels to 2D tensors
        logits_flat = logits.view(-1, self.vocab_size)
        labels_flat = labels.view(-1)

        # Calculate the masked language modeling loss
        loss = self.loss_fn(logits_flat, labels_flat)
        
        return loss
