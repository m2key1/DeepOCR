import random
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch

LINE_END_HYPHENS = '-⸗¬='
GERMAN_LONG_S_BEFORE = frozenset('aäceioöpſstuüyzAÄCEIOÖPSTUÜYZ')
# What a corpus's images show for a flag its GT does not use: 'absent' means the feature never
# appears, 'printed' means it does but no restore rule is trusted, and a restorer name means it
# does and that rule rebuilds it.
IMAGE_STATES = ('absent', 'printed')
# <stem>.diplomatic.txt: the line with every mark its GT normalizes read back from the image
DIPLOMATIC = 'diplomatic'


@dataclass(frozen=True)
class Convention:
    name: str
    marked: str
    plain: str
    normalize: Optional[Callable[[str], str]] = None
    restorers: dict = field(default_factory=dict)


def _restore_long_s(text: str, long_before: Optional[frozenset] = None) -> str:
    chars = list(text)
    for i, char in enumerate(text):
        if char != 's':
            continue
        rest = text[i + 1:].strip()
        # a word broken at the line end continues, so its s is not final
        if len(rest) == 1 and rest in LINE_END_HYPHENS:
            chars[i] = 'ſ'
            continue
        following = text[i + 1] if i + 1 < len(text) else ''
        if following.isalpha() and (long_before is None or following in long_before):
            chars[i] = 'ſ'
    return ''.join(chars)


def restore_long_s_by_position(text: str) -> str:
    return _restore_long_s(text)


def restore_long_s_german(text: str) -> str:
    return _restore_long_s(text, GERMAN_LONG_S_BEFORE)


def normalize_long_s(text: str) -> str:
    return text.replace('ſ', 's')


def normalize_oblique_hyphen(text: str) -> str:
    return text.replace('⸗', '-')


TYPOGRAPHIC_TO_PLAIN = str.maketrans({'’': "'", '‘': "'", '”': '"', '–': '-'})


def normalize_typographic_punctuation(text: str) -> str:
    return text.translate(TYPOGRAPHIC_TO_PLAIN)


# Restorers and normalizers are module-level functions so spawn-context workers can pickle them.
CONVENTIONS = {convention.name: convention for convention in (
    Convention('long_s', 'ſ', 's', normalize_long_s,
               {'position': restore_long_s_by_position, 'german': restore_long_s_german}),
    Convention('oblique_hyphen', '⸗', '-', normalize_oblique_hyphen),
    Convention('typographic_punctuation', '’‘”–', "'\"-", normalize_typographic_punctuation),
    # no rule writes abbreviations out, so this flag only flips on lines that ship the full text
    Convention('expand_abbreviations', '', ''),
)}


class ConventionTable:
    """Per-line convention flags, and the augmentation that flips them. With flip_prob p each
    flag of a sample flips with probability p, rewriting the target to match: a diplomatic line
    is normalized, a normalized line is kept or restored when its corpus's images allow it, a line
    that ships its diplomatic reading trains like a diplomatic line, and a line that ships its
    target for a flag (alternates) can switch to it. Flags missing from
    CONVENTIONS, such as ones a training app defines, flip only through alternates."""

    def __init__(self, names: Sequence[str], by_dataset: dict, image_states: Optional[dict] = None,
                 flip_prob: float = 0.0):
        self.names = list(names)
        self.by_dataset = dict(by_dataset)
        self.image_states = {name: dict(states) for name, states in (image_states or {}).items()}
        self.flip_prob = flip_prob
        for name, states in self.image_states.items():
            known = IMAGE_STATES + tuple(CONVENTIONS[name].restorers if name in CONVENTIONS else ())
            for label, state in states.items():
                if state not in known:
                    raise ValueError(f"{name} state '{state}' for '{label}' is not one of {known}")

    def flags(self, label: str) -> list[float]:
        return [float(value) for value in self.by_dataset.get(label, [0.0] * len(self.names))]

    def sample(self, label: str, text: str, alternates: Optional[dict] = None,
               rng: random.Random = random) -> tuple[str, list[float]]:
        flags = self.flags(label)
        if not self.flip_prob:
            return text, flags
        alternates = alternates or {}
        restored = set()
        if DIPLOMATIC in alternates:
            # the diplomatic reading carries every mark, so it trains like a diplomatic corpus
            text = alternates[DIPLOMATIC]
            restored = {i for i, name in enumerate(self.names)
                        if name in CONVENTIONS and CONVENTIONS[name].normalize is not None and not flags[i]}
            for i in restored:
                flags[i] = 1.0
        decided = set()
        # an alternate is written from the GT, so it is chosen before other flags transform the text
        for i, name in enumerate(self.names):
            if not flags[i] and name in alternates:
                decided.add(i)
                if rng.random() < self.flip_prob:
                    text, flags[i] = alternates[name], 1.0
                    for j in restored:
                        flags[j] = 0.0
                    restored = set()
        for i, name in enumerate(self.names):
            convention = CONVENTIONS.get(name)
            if i in decided or convention is None or rng.random() >= self.flip_prob:
                continue
            if flags[i]:
                if convention.normalize is not None:
                    text, flags[i] = convention.normalize(text), 0.0
                continue
            state = self.image_states.get(name, {}).get(label)
            if state == 'absent':
                flags[i] = 1.0
            elif state in convention.restorers:
                text, flags[i] = convention.restorers[state](text), 1.0
        return text, flags

    def unseen_labels(self, seen: set) -> dict[str, list[str]]:
        """Image-state labels that match no training sample, which is almost always a typo."""
        unseen = {name: sorted(set(states) - seen) for name, states in self.image_states.items()
                  if name in self.names}
        return {name: labels for name, labels in unseen.items() if labels}


def remap_convention_columns(weight: torch.Tensor, old_names: Sequence[str],
                             new_names: Sequence[str]) -> torch.Tensor:
    """Carries a convention projection across a changed flag list by name. A new flag gets a
    random column as long as the trained ones: a flag that is off contributes nothing, so the
    model behaves as before, and one that is on starts with a signal it cannot average away."""
    old_column = {name: i for i, name in enumerate(old_names)}
    remapped = weight.new_zeros(weight.shape[0], len(new_names))
    trained_norm = weight.norm(dim=0).mean() if weight.shape[1] else weight.new_zeros(())
    for new_index, name in enumerate(new_names):
        if name in old_column:
            remapped[:, new_index] = weight[:, old_column[name]]
        else:
            direction = torch.randn(weight.shape[0], dtype=weight.dtype, device=weight.device)
            remapped[:, new_index] = direction / direction.norm() * trained_norm
    return remapped


def hold_trained_flags_to_rate(optimizer: torch.optim.Optimizer, weight: torch.Tensor,
                               names: Sequence[str], new_names: Callable[[], Sequence[str]],
                               slow_group: dict, fast_group: dict):
    """The projection is one tensor in the fast group, but only new flags need the fast rate.
    After each step the trained columns' update is scaled back to the slow group's rate; new_names
    is read per step because a resumed run restores it after the optimizer is built."""
    weight_before_step = {}

    def remember(optimizer, args, kwargs):
        weight_before_step['weight'] = weight.detach().clone()

    def slow_down_trained(optimizer, args, kwargs):
        before = weight_before_step.pop('weight')
        new = set(new_names())
        trained = [index for index, name in enumerate(names) if name not in new]
        if not trained:
            return
        rate_ratio = slow_group['lr'] / fast_group['lr'] if fast_group['lr'] else 0.0
        with torch.no_grad():
            old = before[:, trained]
            weight[:, trained] = old + (weight[:, trained] - old) * rate_ratio

    optimizer.register_step_pre_hook(remember)
    optimizer.register_step_post_hook(slow_down_trained)
