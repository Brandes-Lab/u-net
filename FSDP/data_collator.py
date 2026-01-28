"""
Custom Data Collator for Protein MLM.

This collator properly handles character-level protein tokenizers,
unlike the HuggingFace DataCollatorForLanguageModeling which is
designed for subword tokenizers.
"""

import torch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set


@dataclass
class ProteinMLMCollator:
    """
    Custom MLM collator for protein sequences.
    
    Properly handles:
    - Character-level protein tokenizers
    - Special token masking avoidance
    - Correct label generation for MLM
    
    Args:
        tokenizer: The tokenizer (must have mask_token_id, pad_token_id)
        mlm_probability: Probability of masking a token (default: 0.15)
        special_token_ids: Set of token IDs to never mask (default: uses tokenizer's special tokens)
    """
    tokenizer: Any
    mlm_probability: float = 0.15
    special_token_ids: Optional[Set[int]] = None
    
    def __post_init__(self):
        # Get special token IDs from tokenizer if not provided
        if self.special_token_ids is None:
            if hasattr(self.tokenizer, 'all_special_ids'):
                self.special_token_ids = set(self.tokenizer.all_special_ids)
            else:
                # Default special tokens for protein tokenizers: PAD, UNK, CLS, SEP, MASK
                self.special_token_ids = {0, 1, 2, 3, 4}
        
        # Ensure mask_token_id and pad_token_id exist
        if not hasattr(self.tokenizer, 'mask_token_id') or self.tokenizer.mask_token_id is None:
            raise ValueError("Tokenizer must have mask_token_id")
        if not hasattr(self.tokenizer, 'pad_token_id') or self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have pad_token_id")
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate examples into a batch with MLM masking.
        
        Args:
            examples: List of dicts with 'input_ids' (and optionally 'attention_mask')
            
        Returns:
            Dict with 'input_ids', 'attention_mask', 'labels'
        """
        # Extract input_ids from examples
        if isinstance(examples[0], dict):
            input_ids_list = [torch.tensor(e["input_ids"], dtype=torch.long) for e in examples]
        else:
            # Handle case where examples are just tensors/lists
            input_ids_list = [torch.tensor(e, dtype=torch.long) for e in examples]
        
        # Pad sequences to max length in batch
        max_len = max(len(ids) for ids in input_ids_list)
        
        padded_input_ids = []
        attention_masks = []
        
        for ids in input_ids_list:
            seq_len = len(ids)
            padding_length = max_len - seq_len
            
            if padding_length > 0:
                padded = torch.cat([
                    ids, 
                    torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long)
                ])
                mask = torch.cat([
                    torch.ones(seq_len, dtype=torch.long),
                    torch.zeros(padding_length, dtype=torch.long)
                ])
            else:
                padded = ids
                mask = torch.ones(seq_len, dtype=torch.long)
            
            padded_input_ids.append(padded)
            attention_masks.append(mask)
        
        input_ids = torch.stack(padded_input_ids)
        attention_mask = torch.stack(attention_masks)
        
        # Create masked inputs and labels
        masked_input_ids, labels = self._mask_tokens(input_ids, attention_mask)
        
        return {
            "input_ids": masked_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def _mask_tokens(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> tuple:
        """
        Apply MLM masking to input_ids.
        
        Following BERT:
        - 15% of tokens are selected for prediction
        - Of those: 80% -> [MASK], 10% -> random, 10% -> unchanged
        
        Args:
            input_ids: [B, L] token IDs
            attention_mask: [B, L] attention mask (1 for real tokens, 0 for padding)
            
        Returns:
            masked_input_ids: [B, L] input with masks applied
            labels: [B, L] labels (-100 for non-masked positions)
        """
        labels = input_ids.clone()
        
        # Identify maskable positions (not padding, not special tokens)
        maskable = attention_mask.bool().clone()
        for special_id in self.special_token_ids:
            maskable &= (input_ids != special_id)
        
        # Random mask selection with given probability
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)
        probability_matrix[~maskable] = 0.0
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Set labels to -100 for non-masked positions (ignored in loss)
        labels[~masked_indices] = -100
        
        # 80% of masked tokens -> [MASK]
        indices_replaced = torch.bernoulli(
            torch.full(input_ids.shape, 0.8)
        ).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id
        
        # 10% of masked tokens -> random token (from non-special tokens)
        indices_random = torch.bernoulli(
            torch.full(input_ids.shape, 0.5)  # 0.5 of remaining 20% = 10% total
        ).bool() & masked_indices & ~indices_replaced
        
        # Generate random tokens avoiding special tokens
        # Assuming special tokens are 0-4 and vocab is small (32 for proteins)
        min_token = max(self.special_token_ids) + 1 if self.special_token_ids else 5
        random_tokens = torch.randint(
            min_token, 
            self.tokenizer.vocab_size, 
            input_ids.shape, 
            dtype=torch.long
        )
        input_ids[indices_random] = random_tokens[indices_random]
        
        # Remaining 10% -> unchanged (already correct in input_ids)
        
        return input_ids, labels