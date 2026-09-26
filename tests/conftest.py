import torch
import pytest

@pytest.fixture(autouse=True, scope="session")
def one_thread():
    torch.set_num_threads(1)
