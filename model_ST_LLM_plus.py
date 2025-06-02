import torch
import pickle
import torch.nn as nn
import torch.nn.functional as Fuct
import numpy as np
from transformers import GPT2Model
from typing import Optional, Tuple, Union
from peft import LoraConfig, get_peft_model

class TemporalEmbedding(nn.Module):
    def __init__(self, time, features): 
        super(TemporalEmbedding, self).__init__()
        # This line creates a learnable embedding matrix for each time slot in a day, 
        # so the model can learn to represent time-of-day information in a flexible, trainable way.
        self.time = time  # time：表示一天中的时间步数 features:每个时间步的嵌入是一个 256 维的向量
        self.time_day = nn.Parameter(torch.empty(time, features))  # Registers tensor as a parameter to be optimized
        nn.init.xavier_uniform_(self.time_day)
        # nn.Parameter(...): Tells PyTorch that this tensor should be tracked as a parameter of the model and updated during backpropagation.
        # This creates a learnable parameter (a tensor) called time_week that will be optimized during training 
        self.time_week = nn.Parameter(torch.empty(7, features))  
        nn.init.xavier_uniform_(self.time_week)  # 初始化 self.time_week 参数

    def forward(self, x):  # torch.Size([64, 12, 250, 3])
        day_emb = x[..., 1]  # torch.Size([64, 12, 250]) 从输入数据 x 中提取“星期几”的特征向量
        time_day = self.time_day[  # day_emb[:, -1, :]: Selects the last time step for all batches and all nodes.
            (day_emb[:, -1, :] * self.time).type(torch.LongTensor)  # self.time: The number of time slots in a day (e.g., 288 for 5-min intervals, 48 for 30-min intervals).
        ] # Adds a singleton dimension at the end, so shape becomes [batch, features, nodes, 1].
        time_day = time_day.transpose(1, 2).unsqueeze(-1)  # torch.Size([64, 256, 250, 1])

        week_emb = x[..., 2]  # [64, 12, 250]
        time_week = self.time_week[  # [7, 256]
            (week_emb[:, -1, :]).type(torch.LongTensor)
        ]  # [64, 250, 256]
        time_week = time_week.transpose(1, 2).unsqueeze(-1)  # [64, 256, 250, 1]

        tem_emb = time_day + time_week
        return tem_emb

from dataclasses import dataclass

@dataclass
class BaseModelOutputWithPastAndCrossAttentions:
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class PFA(nn.Module):
    def __init__(self, device="cuda:0", gpt_layers=6, U=1, dropout_rate=0.0):
        super(PFA, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained("gpt2", attn_implementation="eager",
                                              output_attentions=True, output_hidden_states=True)  #attn_implementation="sdpa" OR "eager"
        
        self.gpt2.h = self.gpt2.h[:gpt_layers]  # 通过切片操作，截取 self.gpt2.h 列表的前 gpt_layers 个元素，即保留前 gpt_layers 层
        # print(self.gpt2)
        self.U = U
        self.device = device
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.lora_rank = 16

        self.lora_config = LoraConfig(
            r=self.lora_rank,  # 较小的秩可以减少参数数量，从而提高微调的效率，但可能会牺牲一些模型性能
            lora_alpha=32, # 16 32  缩放因子，用于控制 LoRA 权重的强度
            lora_dropout=self.dropout_rate,  # 防止 LoRA 层的过拟合
            target_modules=['q_attn','c_attn'],  # LoRA 将应用于 GPT-2 模型中的 q_attn 和 c_attn 模块
            bias="none"
        )

        # Configure LoRA only on the last U layers
        # self.lora_config = LoraConfig(
        #     r=self.lora_rank,
        #     lora_alpha=16,  # or any other hyperparameter specific to LoRA
        #     lora_dropout=0.,  # if you want to add dropout to LoRA #0.05
        #     target_modules=['q_attn','c_attn'],  # Apply LoRA to MLP and attention layers in the last U layers only
        #     bias="none"  # specify whether to train bias parameters
        # )
        self.gpt2 = get_peft_model(self.gpt2, self.lora_config)  # 用于将 LoRA 配置应用到预训练模型上
        # 遍历 GPT-2 模型的隐藏层和层中的参数，根据层的索引和参数的名称，灵活地设置参数的 requires_grad 属性
        for layer_index, layer in enumerate(self.gpt2.h):
            for name, param in layer.named_parameters():
                if layer_index < gpt_layers - self.U:
                    if "ln" in name or "wpe" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    if "mlp" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

        # self.apply_lora_to_u_layers()

    # def apply_lora_to_u_layers(self):
        # self.lora_config = LoraConfig(
        #     r=self.lora_rank,
        #     lora_alpha=16,
        #     lora_dropout=self.dropout_rate,
        #     target_modules=['q_attn','c_attn'],
        #     bias="none"
        # )

        # for layer_index in range(len(self.gpt2.h) - self.U, len(self.gpt2.h)):
        #     layer = self.gpt2.h[layer_index]
        #     for name, param in layer.named_parameters():
        #         if "q_attn" in name or "c_attn" in name:
        #             layer = get_peft_model(layer, self.lora_config)

    # Define a custom forward function where the attention_mask.view() step is skipped
    def custom_forward(self,
                    input_ids: Optional[torch.LongTensor] = None,
                    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                    attention_mask: Optional[torch.FloatTensor] = None,
                    token_type_ids: Optional[torch.LongTensor] = None,
                    position_ids: Optional[torch.LongTensor] = None,
                    head_mask: Optional[torch.FloatTensor] = None,
                    inputs_embeds: Optional[torch.FloatTensor] = None,
                    encoder_hidden_states: Optional[torch.Tensor] = None,
                    encoder_attention_mask: Optional[torch.FloatTensor] = None,
                    use_cache: Optional[bool] = None,
                    output_attentions: Optional[bool] = None,
                    output_hidden_states: Optional[bool] = None,
                    return_dict: Optional[bool] = None,
                    adjacency_matrix: Optional[torch.FloatTensor] = None, 
                    ) -> Union[Tuple, dict]:

        output_attentions = output_attentions if output_attentions is not None else self.gpt2.config.output_attentions  # True
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.gpt2.config.output_hidden_states  # True
        use_cache = use_cache if use_cache is not None else self.gpt2.config.use_cache
        return_dict = return_dict if return_dict is not None else self.gpt2.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:  # here
            input_shape = inputs_embeds.size()[:-1]  # torch.Size([64, 250])
            batch_size = inputs_embeds.shape[0]  # 64
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.gpt2.h))  # cache the key and value tensors from previous forward passes
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            # 生成一个从 past_length 开始、到 past_length + seq_len - 1 结束的连续整数序列
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device) 
            position_ids = position_ids.unsqueeze(0)  # 将形状从 (seq_len,) 变为 (1, seq_len)，以便后续广播到整个批次

        if inputs_embeds is None:
            inputs_embeds = self.gpt2.wte(input_ids)
        position_embeds = self.gpt2.wpe(position_ids)  # GPT-2 模型的位置嵌入层（Word Position Embeddings, wpe） 为输入的 position_ids 生成对应的位置嵌入向量
        hidden_states = inputs_embeds + position_embeds  # torch.Size([64, 250, 768])

        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        presents = () if use_cache else None

        total_layers = len(self.gpt2.h)

        for i, (block, layer_past) in enumerate(zip(self.gpt2.h, past_key_values)):
            if i >= total_layers - self.U and adjacency_matrix is not None:  # 控制解冻的层数（如 U=3 表示仅最后 3 层处理图结构）
                attention_mask = adjacency_matrix.to(hidden_states.device).float()
                # attention_mask = attention_mask.to(hidden_states.device) + adjacency_matrix.to(hidden_states.device).float()
            elif attention_mask is not None:  # 若不满足图结构注入条件，则使用原始 attention_mask（如处理填充 token 的掩码）
                attention_mask = attention_mask.to(hidden_states.device)
            # Transformer 层前向传播
            outputs = block(
                hidden_states,
                layer_past=layer_past,  # 缓存的键值对
                attention_mask=attention_mask,  # 图结构或原始掩码
                head_mask=head_mask[i] if head_mask is not None else None,  # 多头掩码
                use_cache=use_cache,  # 是否缓存键值对
                output_attentions=output_attentions,  # 是否返回注意力权重
            )
            hidden_states = outputs[0]

            if use_cache:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2],)
        
        hidden_states = self.gpt2.ln_f(hidden_states)  # Applies the final layer normalization (ln_f) from the GPT-2 model to hidden_states
        # (-1,) means the first dimension (often batch size) is inferred automatically.
        hidden_states = hidden_states.view((-1,) + input_shape[1:] + (hidden_states.size(-1),))  

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions
        )

    def forward(self, x, adjacency_matrix):
        """
        Args:
            x: input embeddings [batch_size, sequence_length, hidden_dim]
            adjacency_matrix: adjacency matrix used as an attention mask
                              [batch_size, sequence_length, sequence_length]  torch.Size([250, 250])
        """
        batch_size =  x.shape[0]  # 64
        num_heads =  self.gpt2.config.n_head  # 12
        adjacency_matrix = adjacency_matrix.unsqueeze(0).repeat(batch_size, 1, 1)  # torch.Size([64, 250, 250]) Now you have one adjacency matrix per sample in the batch.
        adjacency_matrix = adjacency_matrix.unsqueeze(1).repeat(1, num_heads, 1, 1)  # torch.Size([64, 12, 250, 250]) 
        # The final result is a tensor where each batch and each attention head gets its own copy of the adjacency matrix, 
        # matching the expected input shape for multi-head attention layers.
        attention_mask = adjacency_matrix.to(self.device).float() #[64,12,250,250]
        # print(attention_mask.shape)  # torch.Size([64, 12, 250, 250])

        # Use GPT-2 with attention mask
        output = self.custom_forward(
            inputs_embeds=x,
            attention_mask=attention_mask
        ).last_hidden_state
        output = self.dropout(output)

        return output


class ST_LLM(nn.Module):
    def __init__(
        self,
        device,
        adj_mx,
        input_dim=3,
        num_nodes=170,
        input_len=12,
        output_len=12,
        llm_layer=6,
        U=1
    ):
        super().__init__()  # 调用父类（或超类）的构造方法

        self.device = device
        self.adj_mx = torch.tensor(adj_mx, dtype=torch.float32).to(self.device)        
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.llm_layer = llm_layer
        self.U = U
        self.dropout_rate = 0.1
        
        if num_nodes == 170 or num_nodes == 207:
            time = 288 # 288 = 24 hours × 12 intervals/hour = 5 minutes per interval.
        elif num_nodes == 250 or num_nodes == 266:
            time = 48  # 48 = 24 hours × 2 intervals/hour = 30 minutes per interval.

        elif num_nodes == 275:  # evdata
            time = 24

        gpt_channel = 256  # 输出数据的通道数，即卷积层的目标特征维度
        to_gpt_channel = 768  # GPT-2 模型的标准隐藏层维度
        # 用于将输入数据的特征维度从 self.input_dim * self.input_len 转换为 gpt_channel
        self.start_conv = nn.Conv2d(
            self.input_dim * self.input_len, gpt_channel, kernel_size=(1, 1)
        )

        self.Temb = TemporalEmbedding(time, gpt_channel)  
        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, gpt_channel))  # 用于表示图中每个节点的嵌入
        nn.init.xavier_uniform_(self.node_emb)
        # kernel_size=(1, 1) 点卷积（pointwise convolution），它不会改变输入数据的空间维度（高度和宽度），但会改变通道数。 
        self.in_layer = nn.Conv2d(gpt_channel*3, to_gpt_channel, kernel_size=(1, 1))  # 维度调整
        self.dropout = nn.Dropout(p=self.dropout_rate)         

        # regression
        self.regression_layer = nn.Conv2d(to_gpt_channel, self.output_len, kernel_size=(1, 1)) 

        # GPT2
        self.gpt = PFA(device=self.device, gpt_layers=self.llm_layer, U=self.U, dropout_rate=self.dropout_rate)
                 
    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])
    
    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, history_data):  # torch.Size([64, 3, 250, 12])
        # rearranges the tensor so that the features dimension comes second, resulting in a shape of [batch, features, nodes, time] This prepares the data for processing by models that expect the features as channels
        data = history_data.permute(0, 3, 2, 1)  # torch.Size([64, 12, 250, 3])
        B, T, S, F = data.shape
        # print(data.shape) #[64, 12, 250, 3]
        
        #Temporal Embedding # 这个过程是论文里的 Ed T = Wd(Xday) Ew T = Ww(Xweek),ET = Ed T + Ew T
        tem_emb = self.Temb(data)  # TemporalEmbedding.forward(data)
        # print(tem_emb.shape) #[64, 256, 250, 1]
        # ES = σ(WS · XP + bs)
        node_emb = []
        node_emb.append( # The goal is to align both the temporal and node embeddings so they can be concatenated or combined with other features along the "channel" (256) dimension for further processing (e.g., convolutional layers).
            self.node_emb.unsqueeze(0)  # Adds a batch dimension at the front, shape becomes [1, 250, 256].
            .expand(B, -1, -1)  # Copies the node embeddings for each batch. Shape now [B, 250, 256]
            .transpose(1, 2)  # Swaps axis 1 and 2, shape becomes [B, 256, 250].
            .unsqueeze(-1)  # Adds a singleton dimension at the end, shape becomes [B, 256, 250, 1].
        )
        # data.shape [64, 12, 250, 3]
        input_data = data.permute(0,3,2,1) #[32, 2, 207, 12] torch.Size([64, 3, 250, 12])
        input_data = input_data.transpose(1, 2).contiguous() #[32, 207, 2, 12]  torch.Size([64, 250, 3, 12])
        input_data = (input_data.view(B, S, -1).transpose(1, 2).unsqueeze(-1))
        # print(input_data.shape) #[64, 36, 250, 1]  torch.Size([64, 36, 250, 1])
        input_data = self.start_conv(input_data)  # torch.Size([64, 256, 250, 1])
        # print(input_data.shape)  #[64, 36, 250, 1]
        # HF = FConv (EP ||ES||ET ; θf )
        data_st = torch.cat([input_data] + [tem_emb] + node_emb, dim=1)
        # print(f"After cat: data_st shape: {data_st.shape}, type: {type(data_st)}")

        data_st = self.in_layer(data_st)  # torch.Size([64, 768, 250, 1])
        # print(f"After in_layer: data_st shape: {data_st.shape}, type: {type(data_st)}")

        data_st = Fuct.leaky_relu(data_st)  # torch.Size([64, 768, 250, 1])
        data_st = data_st.permute(0, 2, 1, 3).squeeze(-1)  # torch.Size([64, 250, 768])
        
        outputs = self.gpt(data_st, self.adj_mx)  # torch.Size([64, 250, 768])
            
        outputs = outputs.permute(0, 2, 1).unsqueeze(-1)
        # print(outputs.shape) # torch.Size([64, 768, 250, 1]) 

        # regression It applies a linear transformation (optionally with bias) to map from the model’s internal feature space to the desired output size.
        outputs = self.regression_layer(outputs)  
        # print(outputs.shape) #[64, 12, 250, 1]
        
        return outputs
