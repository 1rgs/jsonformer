from transformers import PreTrainedTokenizer, LogitsWarper, StoppingCriteria
import torch


class NumberStoppingCriteria(StoppingCriteria):
    """
    This class can be used to stop generation when there is a repeated decimal point in the generated text.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, precision: int = 2):
        self.tokenizer = tokenizer
        self.precision = precision

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> bool:
        decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if ".." in decoded:
            print("Stopping because of ..")
            return True

        if decoded.strip().count(".") > 1:
            print("Stopping because of multiple .")
            return True

        if (
            decoded.strip().count(".") == 1
            and len(decoded.strip().split(".")[1]) > self.precision
        ):
            return True

        return False


class OutputNumbersTokens(LogitsWarper):
    def __init__(self, tokenizer: PreTrainedTokenizer, prompt: str):
        self.whitelist_tokens = [tokenizer.eos_token_id]
        self.tokenized_prompt = tokenizer(prompt, return_tensors="pt")

        for token_str, token_id in tokenizer.get_vocab().items():
            if (
                (
                    token_str.startswith("Ġ")
                    and (
                        all(c.isdigit() or c == "." for c in token_str[1:])
                        and token_str.count(".") <= 1
                    )
                )
                or (
                    all(c.isdigit() or c == "." for c in token_str)
                    and token_str.count(".") <= 1
                )
                or (
                    token_str[-1] == " "
                    and all(c.isdigit() or c == "." for c in token_str[:-1])
                )
            ):
                self.whitelist_tokens.append(token_id)

    def __call__(self, input_ids, scores):
        input_ids = input_ids[:, len(self.tokenized_prompt["input_ids"][0]) :]
        scores[
            :, [i for i in range(len(scores[0])) if i not in self.whitelist_tokens]
        ] = -float("inf")
        return scores
