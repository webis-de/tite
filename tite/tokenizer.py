from transformers import BertTokenizerFast


class TiteTokenizer(BertTokenizerFast):

    def __init__(
        self,
        vocab_file: str,
        tokenizer_file: str,
        do_lower_case: bool = True,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        tokenize_chinese_chars: bool = True,
        strip_accents: bool | None = None,
        **kwargs
    ):
        super().__init__(
            vocab_file,
            tokenizer_file,
            do_lower_case,
            unk_token,
            sep_token,
            pad_token,
            cls_token,
            mask_token,
            tokenize_chinese_chars,
            strip_accents,
            **kwargs
        )
