#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: di.wu@mobvoi.com (DI WU)
"""Encoder self-attention layer definition."""

from typing import Optional, Tuple

import torch
from torch import nn

import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import numpy as np
import random

import math
import torch.nn.functional as F

import time

class TransformerEncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: to use layer_norm after each sub-block.
        concat_after (bool): Whether to concat attention layer's input and
            output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)

    """
    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: torch.nn.Module,
        dropout_rate: float,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size, eps=1e-12)
        self.norm2 = nn.LayerNorm(size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        # concat_linear may be not used in forward fuction,
        # but will be saved in the *.pt
        self.concat_linear = nn.Linear(size + size, size)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: Optional[torch.Tensor] = None,
        output_cache: Optional[torch.Tensor] = None,
        cnn_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            pos_emb (torch.Tensor): just for interface compatibility
                to ConformerEncoderLayer
            mask_pad (torch.Tensor): does not used in transformer layer,
                just for unified api with conformer.
            output_cache (torch.Tensor): Cache tensor of the output
                (#batch, time2, size), time2 < time in x.
            cnn_cache (torch.Tensor): not used here, it's for interface
                compatibility to ConformerEncoderLayer
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if output_cache is None:
            x_q = x
        else:
            assert output_cache.size(0) == x.size(0)
            assert output_cache.size(2) == self.size
            assert output_cache.size(1) < x.size(1)
            chunk = x.size(1) - output_cache.size(1)
            x_q = x[:, -chunk:, :]
            residual = residual[:, -chunk:, :]
            mask = mask[:, -chunk:, :]

        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x_q, x, x, mask)), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(self.self_attn(x_q, x, x, mask))

        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        if output_cache is not None:
            x = torch.cat([output_cache, x], dim=1)

        fake_cnn_cache = torch.tensor([0.0], dtype=x.dtype, device=x.device)
        return x, mask, fake_cnn_cache


class ConformerEncoderLayer(nn.Module):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module
             instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: use layer_norm after each sub-block.
        concat_after (bool): Whether to concat attention layer's input and
            output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """
    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: Optional[nn.Module] = None,
        feed_forward_macaron: Optional[nn.Module] = None,
        conv_module: Optional[nn.Module] = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(size, eps=1e-12)  # for the FNN module
        self.norm_mha = nn.LayerNorm(size, eps=1e-12)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-12)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = nn.LayerNorm(size,
                                          eps=1e-12)  # for the CNN module
            self.norm_final = nn.LayerNorm(
                size, eps=1e-12)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        self.concat_linear = nn.Linear(size + size, size)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: Optional[torch.Tensor] = None,
        output_cache: Optional[torch.Tensor] = None,
        cnn_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, time，time).
            pos_emb (torch.Tensor): positional encoding, must not be None
                for ConformerEncoderLayer.
            mask_pad (torch.Tensor): batch padding mask used for conv module.
                (#batch, 1，time)
            output_cache (torch.Tensor): Cache tensor of the output
                (#batch, time2, size), time2 < time in x.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).
        """

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(
                self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        if output_cache is None:
            x_q = x
        else:
            assert output_cache.size(0) == x.size(0)
            assert output_cache.size(2) == self.size
            assert output_cache.size(1) < x.size(1)
            chunk = x.size(1) - output_cache.size(1)
            x_q = x[:, -chunk:, :]
            residual = residual[:, -chunk:, :]
            mask = mask[:, -chunk:, :]

        x_att = self.self_attn(x_q, x, x, mask, pos_emb)
        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = torch.tensor([0.0], dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)

            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        if output_cache is not None:
            x = torch.cat([output_cache, x], dim=1)

        return x, mask, new_cnn_cache

def chunk_partition(
        x : torch.Tensor, 
        chunk_size : int
        )->torch.Tensor:
    """
    Args:
        x: (batch_size, sequence_len, channel_dim)
        chunk_size (int): chunk size
    Returns:
        chunks: (num_chunks*batch_size, chunk_size, channel_dim)
    """
    #B batch size , W sample size, C channel->(Fbank)
    B, W, C = x.shape
    #partition along height and width
    x = x.view(B, W // chunk_size, chunk_size, C)
    chunks = x.contiguous().view(-1, chunk_size, C)

    return chunks

def chunk_reverse(
        chunks: torch.Tensor,
        chunk_size : int,
        sequence_len: int
        )->torch.Tensor:
    """
    Args:
        chunks: (num_chunks*batch_size, chunk_size, channel_dim)
        chunk_size (int): chunk size
        sequence_len(int): len of wav feature sequence

    Returns:
        x: (batch_size, sequence_len, channel_dim)
    """
    B = int(chunks.shape[0] / ( sequence_len / chunk_size))
    x = chunks.view(B, sequence_len, -1)
    return x

class SChunkTransformerEncoderLayer(nn.Module):
    r""" SChunk Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        chunk_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
            self, 
            size: int, 
            self_attn: torch.nn.Module,
            feed_forward: torch.nn.Module,
            dropout_rate: float,
            normalize_before: bool = True,
            concat_after: bool = False,
            input_resolution: int = 384, 
            chunk_size: int = 16,
            shift_size : int = 0,
            ):

        super().__init__()
        self.size = size 
        self.input_resolution = input_resolution
        self.num_heads = 4 
        self.chunk_size = chunk_size
        self.shift_size = shift_size

        if self.input_resolution <= self.chunk_size:
            # if chink size is larger than input resolution, we don't partition chunks 
            self.shift_size = 0
            self.chunk_size = self.input_resolution
        assert 0 <= self.shift_size < self.chunk_size, "shift_size must in 0-chunk_size"

        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size, eps=1e-12)
        self.norm2 = nn.LayerNorm(size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        # concat_linear may be not used in forward fuction,
        # but will be saved in the *.pt
        self.concat_linear = nn.Linear(size + size, size)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor,
        output_cache: Optional[torch.Tensor] = None,
        cnn_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        #W = self.input_resolution
        B, L, C = x.shape
        #print( "421: L = ", L )
        #print( "421: W = ", W )
        #assert L == W, "input feature has wrong size"
        W = L

        if self.shift_size > 0:
            # calculate chunk attention mask for chunk-base MSA
            sequence_len  = W 
            #chunk_mask = torch.zeros((1, sequence_len, 1)) 
            chunk_mask = torch.zeros((1, sequence_len, 1)).cuda() 

            mask_chunks = chunk_partition(chunk_mask, self.chunk_size)  # nW, chunk_size, chunk_size, 1
            mask_chunks = mask_chunks.view(-1, self.chunk_size)
            attn_mask = mask_chunks.unsqueeze(1) - mask_chunks.unsqueeze(2)

            attn_mask[:-1,:int(self.chunk_size/2-1),int(self.chunk_size/2):] = float(-100.0)
            attn_mask[-1:,int(self.chunk_size/2):,int(self.chunk_size/2):] = float(-100.0)
        else:
            attn_mask = torch.ones(1,dtype=torch.int8).cuda()
            #attn_mask = torch.ones(1,dtype=torch.int8)
            #attn_mask = torch.IntTensor([1]).cuda()

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if output_cache is None:
            x_q = x
        else:
            assert output_cache.size(0) == x.size(0)
            assert output_cache.size(2) == self.size
            assert output_cache.size(1) < x.size(1)
            chunk = x.size(1) - output_cache.size(1)
            #x_q = x[:, -chunk:, :]
            #residual = residual[:, -chunk:, :]
            #mask = mask[:, -chunk:, :]
            x_q = x

        shortcut = x

        #print( "encoder_layer.py 427: mask_pad.shape = ", mask_pad.shape )
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size), dims=(1))
            shifted_x_q = torch.roll(x_q, shifts=(-self.shift_size), dims=(1))
            #shifted_mask = torch.roll(mask, shifts=(-self.shift_size), dims=(1))
            shifted_mask = torch.roll(mask_pad.squeeze(1), shifts=(-self.shift_size), dims=(1))
        else:
            shifted_x = x
            shifted_x_q = x_q
            shifted_mask = mask_pad.squeeze(1)
            #shifted_mask = mask

        # partition chunks
        x_chunks = chunk_partition(shifted_x, self.chunk_size)  # nW*B, chunk_size, chunk_size, C
        x_chunks = x_chunks.view(-1, self.chunk_size, C)  # nW*B, window_size*window_size, C

        x_q_chunks = chunk_partition(shifted_x_q, self.chunk_size)  # nW*B, chunk_size, chunk_size, C
        x_q_chunks = x_q_chunks.view(-1, self.chunk_size, C)  # nW*B, window_size*window_size, C

        mask_chunks = chunk_partition( shifted_mask.unsqueeze(-1), self.chunk_size )
        #print( "encoder_layer.py 445: mask_chunks.shape = ", mask_chunks.view( -1, self.chunk_size, 1 ).shape )
        mask_chunks = mask_chunks.view( -1, self.chunk_size, 1 ).squeeze(-1)  # nW*B, window_size, C
        #print( "encoder_layer.py 447: mask_chunks.shape = ", mask_chunks.shape )
        #enc_attn_mask = mask_chunks.unsqueeze(1).expand(-1, self.chunk_size, -1)

        #print( "encoder_layer.py 449: x_q_chunks.shape = ", x_q_chunks.shape, "x_chunks.shape = ", x_chunks.shape )
        attn_chunks = self.self_attn(x_q_chunks, x_chunks, x_chunks,
                                        mask=mask_chunks, chunk_mask = attn_mask )  # nW*B, chunk_size*chunk_size, C
        # merge chunks
        attn_chunks = attn_chunks.view(-1, self.chunk_size, C)
        shifted_x = chunk_reverse(attn_chunks, self.chunk_size, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size), dims=(1))
            mask = torch.roll(shifted_mask, shifts=(self.shift_size), dims=(1))
        else:
            x = shifted_x
            mask = shifted_mask

        x *= mask.unsqueeze(-1)
        mask = mask.squeeze(-1)

        x = x.view(B, W, C)

        if self.concat_after:
            x_concat = torch.cat((shortcut, x ), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x)

        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        #if output_cache is not None:
        #    x = torch.cat([output_cache, x], dim=1)

        fake_cnn_cache = torch.tensor([0.0], dtype=x.dtype, device=x.device)

        return x, mask, fake_cnn_cache

class SChunkConformerEncoderLayer(nn.Module):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module
             instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: use layer_norm after each sub-block.
        concat_after (bool): Whether to concat attention layer's input and
            output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """
    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: Optional[nn.Module] = None,
        feed_forward_macaron: Optional[nn.Module] = None,
        conv_module: Optional[nn.Module] = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
        concat_after: bool = False,
        input_resolution: int = 384, 
        chunk_size: int = 16,
        shift_size : int = 0,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.size = size 
        self.input_resolution = input_resolution
        self.num_heads = 4 
        self.chunk_size = chunk_size
        self.shift_size = shift_size

        if self.input_resolution <= self.chunk_size:
            # if chink size is larger than input resolution, we don't partition chunks 
            self.shift_size = 0
            self.chunk_size = self.input_resolution
        assert 0 <= self.shift_size < self.chunk_size, "shift_size must in 0-chunk_size"

        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(size, eps=1e-12)  # for the FNN module
        self.norm_mha = nn.LayerNorm(size, eps=1e-12)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-12)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = nn.LayerNorm(size,
                                          eps=1e-12)  # for the CNN module
            self.norm_final = nn.LayerNorm(
                size, eps=1e-12)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        self.concat_linear = nn.Linear(size + size, size)

        """
        if self.shift_size > 0:
            # calculate chunk attention mask for chunk-base MSA
            sequence_len  = self.input_resolution
            chunk_mask = torch.zeros((1, sequence_len, 1)) 

            slices = (slice(0, -self.chunk_size),
                        slice(-self.chunk_size, -self.shift_size),
                        slice(-self.shift_size, None))
            #cnt = 0
            #for w in slices:
            #    chunk_mask[:, w, :] = cnt
            #    cnt += 1

            mask_chunks = chunk_partition(chunk_mask, self.chunk_size)  # nW, chunk_size, chunk_size, 1
            mask_chunks = mask_chunks.view(-1, self.chunk_size)
            attn_mask = mask_chunks.unsqueeze(1) - mask_chunks.unsqueeze(2)

            #attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            #attn_mask = torch.flip( attn_mask, [0] )
            attn_mask[:-1,:int(self.chunk_size/2-1),int(self.chunk_size/2):] = float(-100.0)
            attn_mask[-1:,int(self.chunk_size/2):,int(self.chunk_size/2):] = float(-100.0)

            #print( "393: attn_mask = ", attn_mask[0,:,:])
            #print( "393: attn_mask = ", attn_mask[-1,:,:])

        else:
            attn_mask = torch.IntTensor([1])

        self.register_buffer("attn_mask", attn_mask)
        """


    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor,
        output_cache: Optional[torch.Tensor] = None,
        cnn_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, time，time).
            pos_emb (torch.Tensor): positional encoding, must not be None
                for ConformerEncoderLayer.
            mask_pad (torch.Tensor): batch padding mask used for conv module.
                (#batch, 1，time)
            output_cache (torch.Tensor): Cache tensor of the output
                (#batch, time2, size), time2 < time in x.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).
        """
        W = self.input_resolution
        B, L, C = x.shape
        #assert L == W, "input feature has wrong size"
        W = L

        if self.shift_size > 0:
            # calculate chunk attention mask for chunk-base MSA
            sequence_len  = W 
            #chunk_mask = torch.zeros((1, sequence_len, 1)) 
            chunk_mask = torch.zeros((1, sequence_len, 1)).cuda() 

            mask_chunks = chunk_partition(chunk_mask, self.chunk_size)  # nW, chunk_size, chunk_size, 1
            mask_chunks = mask_chunks.view(-1, self.chunk_size)
            attn_mask = mask_chunks.unsqueeze(1) - mask_chunks.unsqueeze(2)

            attn_mask[:-1,:int(self.chunk_size/2-1),int(self.chunk_size/2):] = float(-100.0)
            attn_mask[-1:,int(self.chunk_size/2):,int(self.chunk_size/2):] = float(-100.0)
        else:
            attn_mask = torch.ones(1,dtype=torch.int8).cuda()
            #attn_mask = torch.ones(1,dtype=torch.int8)
            #attn_mask = torch.IntTensor([1]).cuda()

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(
                self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        #print( "encoder_layer.py 659: output_cache = ", output_cache )
        if output_cache is None:
            x_q = x
        else:
            assert output_cache.size(0) == x.size(0)
            assert output_cache.size(2) == self.size
            assert output_cache.size(1) < x.size(1)
            chunk = x.size(1) - output_cache.size(1)
            x_q = x[:, -chunk:, :]
            residual = residual[:, -chunk:, :]
            mask = mask[:, -chunk:, :]

        shortcut = x

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size), dims=(1))
            shifted_x_q = torch.roll(x_q, shifts=(-self.shift_size), dims=(1))
            #shifted_mask = torch.roll(mask, shifts=(-self.shift_size), dims=(1))
            shifted_mask = torch.roll(mask_pad.squeeze(1), shifts=(-self.shift_size), dims=(1))
        else:
            shifted_x = x
            shifted_x_q = x_q
            shifted_mask = mask_pad.squeeze(1)
            #shifted_mask = mask

        # partition chunks
        x_chunks = chunk_partition(shifted_x, self.chunk_size)  # nW*B, chunk_size, chunk_size, C
        x_chunks = x_chunks.view(-1, self.chunk_size, C)  # nW*B, window_size*window_size, C

        x_q_chunks = chunk_partition(shifted_x_q, self.chunk_size)  # nW*B, chunk_size, chunk_size, C
        x_q_chunks = x_q_chunks.view(-1, self.chunk_size, C)  # nW*B, window_size*window_size, C

        mask_chunks = chunk_partition( shifted_mask.unsqueeze(-1), self.chunk_size )
        mask_chunks = mask_chunks.view( -1, self.chunk_size, 1 ).squeeze()  # nW*B, window_size, C
        #enc_attn_mask = mask_chunks.unsqueeze(1).expand(-1, self.chunk_size, -1)

        
        attn_chunks = self.self_attn(x_q_chunks, x_chunks, x_chunks,
                                        mask=mask_chunks, chunk_mask = attn_mask )  # nW*B, chunk_size*chunk_size, C
                                        #mask=mask_chunks, chunk_mask = self.attn_mask )  # nW*B, chunk_size*chunk_size, C
        # merge chunks
        attn_chunks = attn_chunks.view(-1, self.chunk_size, C)
        shifted_x = chunk_reverse(attn_chunks, self.chunk_size, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size), dims=(1))
            mask = torch.roll(shifted_mask, shifts=(self.shift_size), dims=(1))
        else:
            x = shifted_x
            mask = shifted_mask

        x *= mask.unsqueeze(-1)
        mask = mask.squeeze(-1)

        x_att = x.view(B, W, C)

        if self.concat_after:
            x_concat = torch.cat((shortcut, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = torch.tensor([0.0], dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)

            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        if output_cache is not None:
            x = torch.cat([output_cache, x], dim=1)

        return x, mask, new_cnn_cache


