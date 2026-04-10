from __future__ import annotations

from dataclasses import dataclass


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "SET", "=", ";", "QRY", "ANS", "W", "R"]


@dataclass(frozen=True)
class Vocabulary:
    tokens: list[str]
    token_to_id: dict[str, int]
    key_tokens: list[str]
    value_tokens: list[str]

    @classmethod
    def build(cls, num_keys: int, num_values: int) -> "Vocabulary":
        if num_keys <= 0 or num_values <= 0:
            raise ValueError("num_keys and num_values must be positive.")
        key_width = max(2, len(str(num_keys - 1)))
        value_width = max(2, len(str(num_values - 1)))
        key_tokens = [f"K{index:0{key_width}d}" for index in range(num_keys)]
        value_tokens = [f"V{index:0{value_width}d}" for index in range(num_values)]
        tokens = [*SPECIAL_TOKENS, *key_tokens, *value_tokens]
        return cls(
            tokens=tokens,
            token_to_id={token: index for index, token in enumerate(tokens)},
            key_tokens=key_tokens,
            value_tokens=value_tokens,
        )

    @classmethod
    def from_metadata(cls, data: dict[str, object]) -> "Vocabulary":
        tokens = list(data["tokens"])
        key_tokens = list(data["key_tokens"])
        value_tokens = list(data["value_tokens"])
        return cls(
            tokens=tokens,
            token_to_id={token: index for index, token in enumerate(tokens)},
            key_tokens=key_tokens,
            value_tokens=value_tokens,
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "tokens": self.tokens,
            "key_tokens": self.key_tokens,
            "value_tokens": self.value_tokens,
        }

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id["<eos>"]

    def encode(self, tokens: list[str]) -> list[int]:
        try:
            return [self.token_to_id[token] for token in tokens]
        except KeyError as error:
            raise KeyError(f"Unknown token during encoding: {error.args[0]}") from error

    def decode(self, token_ids: list[int]) -> list[str]:
        decoded: list[str] = []
        for token_id in token_ids:
            if token_id < 0 or token_id >= len(self.tokens):
                raise IndexError(f"Token id out of range: {token_id}")
            decoded.append(self.tokens[token_id])
        return decoded

    @property
    def value_token_ids(self) -> list[int]:
        return [self.token_to_id[token] for token in self.value_tokens]
