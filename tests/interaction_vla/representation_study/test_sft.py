from interaction_vla.representation_study.sft import DeterministicStepBatchSampler


def test_sft_sampler_is_resume_exact() -> None:
    full = list(
        DeterministicStepBatchSampler(
            dataset_size=17, batch_size=3, seed=5, start_step=0, total_steps=10
        )
    )
    resumed = list(
        DeterministicStepBatchSampler(
            dataset_size=17, batch_size=3, seed=5, start_step=4, total_steps=10
        )
    )
    assert resumed == full[4:]
