from __future__ import annotations

from autoresearch_rl.mdp import Action, Reward, State
from autoresearch_rl.trajectory import Transition, TrajectoryBuffer


def _state(iteration: int = 0) -> State:
    return State(
        code_hash="abc",
        history=(),
        metrics={"loss": 0.5},
        resource_budget=100.0,
        iteration=iteration,
    )


def _transition(iteration: int = 0, reward_val: float = 1.0) -> Transition:
    return Transition(
        state=_state(iteration),
        action=Action(params={"lr": 0.01}),
        reward=Reward(value=reward_val),
        next_state=_state(iteration + 1),
        log_prob=-0.5,
        value_estimate=0.8,
    )


def test_add_and_len() -> None:
    buf = TrajectoryBuffer(max_size=10)
    assert len(buf) == 0
    buf.add(_transition(0))
    assert len(buf) == 1
    buf.add(_transition(1))
    assert len(buf) == 2


def test_circular_drops_oldest() -> None:
    buf = TrajectoryBuffer(max_size=3)
    for i in range(5):
        buf.add(_transition(i, reward_val=float(i)))
    assert len(buf) == 3
    assert buf.rewards == [2.0, 3.0, 4.0]


def test_get_batch() -> None:
    buf = TrajectoryBuffer(max_size=10)
    for i in range(5):
        buf.add(_transition(i))
    batch = buf.get_batch(3)
    assert len(batch) == 3
    assert batch[0].state.iteration == 2
    assert batch[2].state.iteration == 4


def test_get_episode_full() -> None:
    buf = TrajectoryBuffer(max_size=10)
    for i in range(4):
        buf.add(_transition(i))
    ep = buf.get_episode()
    assert len(ep) == 4


def test_get_episode_slice() -> None:
    buf = TrajectoryBuffer(max_size=10)
    for i in range(5):
        buf.add(_transition(i))
    ep = buf.get_episode(start=1, end=3)
    assert len(ep) == 2
    assert ep[0].state.iteration == 1
    assert ep[1].state.iteration == 2


def test_clear() -> None:
    buf = TrajectoryBuffer(max_size=10)
    buf.add(_transition(0))
    buf.add(_transition(1))
    buf.clear()
    assert len(buf) == 0


def test_rewards_property() -> None:
    buf = TrajectoryBuffer(max_size=10)
    buf.add(_transition(0, reward_val=1.0))
    buf.add(_transition(1, reward_val=2.5))
    buf.add(_transition(2, reward_val=-0.3))
    assert buf.rewards == [1.0, 2.5, -0.3]


def test_values_property() -> None:
    buf = TrajectoryBuffer(max_size=10)
    buf.add(_transition(0))
    assert buf.values == [0.8]


def test_empty_properties() -> None:
    buf = TrajectoryBuffer(max_size=10)
    assert buf.rewards == []
    assert buf.values == []


def test_get_batch_larger_than_buffer() -> None:
    buf = TrajectoryBuffer(max_size=10)
    buf.add(_transition(0))
    batch = buf.get_batch(100)
    assert len(batch) == 1
