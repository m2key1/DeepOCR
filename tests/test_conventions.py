import torch
import torch.nn as nn

from utils.model import DeepOCR, remap_convention_columns


def projection(*columns: float) -> torch.Tensor:
    """A d_model 2 projection whose column i is filled with columns[i], so a column is known by value."""
    return torch.tensor([list(columns), list(columns)])


def test_a_known_flag_keeps_its_column_wherever_it_now_sits():
    base = projection(1.0, 2.0)
    out = remap_convention_columns(base, ['long_s', 'oblique_hyphen'], ['oblique_hyphen', 'long_s'])
    assert out.tolist() == projection(2.0, 1.0).tolist()


def test_a_new_flag_starts_at_zero_beside_the_ones_kept():
    base = projection(1.0, 2.0)
    out = remap_convention_columns(base, ['long_s', 'oblique_hyphen'], ['long_s', 'oblique_hyphen', 'keep_hyphens'])
    assert out.tolist() == projection(1.0, 2.0, 0.0).tolist()


def test_a_flag_dropped_from_the_names_is_dropped_from_the_columns():
    out = remap_convention_columns(projection(1.0, 2.0), ['long_s', 'oblique_hyphen'], ['oblique_hyphen'])
    assert out.tolist() == projection(2.0).tolist()


class Conditioned(nn.Module):
    """Just the part of DeepOCR that a fine-tune's conventions touch."""
    _adopt_base_conventions = DeepOCR._adopt_base_conventions

    def __init__(self, names, by_dataset):
        super().__init__()
        self.d_model = 2
        self.convention_names = list(names)
        self.convention_by_dataset = dict(by_dataset)
        self.convention_proj = nn.Linear(len(names), 2, bias=False) if names else None


def test_a_fine_tune_given_no_flags_keeps_its_bases():
    model = Conditioned([], {})
    model._adopt_base_conventions(['long_s', 'oblique_hyphen'])
    assert model.convention_names == ['long_s', 'oblique_hyphen']
    assert model.convention_proj.weight.shape == (2, 2)
    assert model.convention_proj.weight.abs().sum().item() == 0


def test_a_fine_tune_teaching_a_new_flag_puts_its_own_first_and_pads_its_rows():
    model = Conditioned(['long_s', 'keep_hyphens'], {'job1': [1.0, 1.0], 'job2': [0.0, 0.0]})
    model._adopt_base_conventions(['long_s', 'oblique_hyphen'])
    assert model.convention_names == ['long_s', 'keep_hyphens', 'oblique_hyphen']
    assert model.convention_by_dataset == {'job1': [1.0, 1.0, 0.0], 'job2': [0.0, 0.0, 0.0]}
    assert model.convention_proj.weight.shape == (2, 3)


def test_a_fine_tune_that_already_names_every_base_flag_is_left_alone():
    model = Conditioned(['long_s'], {'job1': [1.0]})
    proj = model.convention_proj
    model._adopt_base_conventions(['long_s'])
    assert model.convention_proj is proj
