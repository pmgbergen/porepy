"""Tests for the parameters dictionary initialization."""

import pytest

import porepy as pp


@pytest.mark.parametrize("empty_initial_data", [True, False])
@pytest.mark.parametrize("with_grid", [True, False])
@pytest.mark.parametrize("specified_parameters_type", ["none", "args", "kwargs"])
def test_initialize_parameters(
    with_grid: bool, specified_parameters_type: str, empty_initial_data: bool
):
    """Tests old and new signatures of `pp.initial_parameters`. Parametrization
    `with_grids` should be removed when deprecating the old signature. So far, making
    sure that it emits a warning and works as expected.

    """
    keyword = "flow"  # Random keyword.

    # Constructing the arguments for the method.
    if empty_initial_data:
        initial_data = {}
    else:
        # This ensures that we do not lose what was present in the dictionary before.
        initial_data = {pp.PARAMETERS: {keyword: {"initial_arg": 42}}}
    args: list = [initial_data, keyword]

    # A warning will be emitted if using old signature.
    should_warn = False
    if with_grid:
        should_warn = True
        # The "grid" argument is not used, so we can pass anything there.
        args = ["dummy_grid"] + args

    # 3 cases for the optional argument: passing it in args (positional), kwargs (named)
    # or not passing it.
    kwargs = {}
    specified_parameters = {"new_arg": 1234}
    if specified_parameters_type == "args":
        args.append(specified_parameters)
    elif specified_parameters_type == "kwargs":
        kwargs["specified_parameters"] = specified_parameters

    if should_warn:
        with pytest.warns():
            result = pp.initialize_data(*args, **kwargs)
    else:
        result = pp.initialize_data(*args, **kwargs)

    expected = {
        pp.PARAMETERS: {
            keyword: ({} if empty_initial_data else {"initial_arg": 42})
            | (specified_parameters if specified_parameters_type != "none" else {}),
        },
        pp.DISCRETIZATION_MATRICES: {keyword: {}},
    }
    assert result == expected
