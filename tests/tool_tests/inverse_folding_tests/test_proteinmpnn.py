"""
test_proteinmpnn.py

Tests for ProteinMPNN inverse folding tools.
"""

import json
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest

from bio_programming.tools.inverse_folding.proteinmpnn.proteinmpnn import (
    MPNN_ALPHABET,
    ProteinMPNNScoringConfig,
    ProteinMPNNScoringInput,
    SequenceStructurePair,
    run_proteinmpnn_sample,
    run_proteinmpnn_score,
)
from bio_programming.tools.language_models.schemas import SequenceScores
from bio_programming.tools.inverse_folding.schemas import (
    InverseFoldingConfig,
    InverseFoldingInput,
    InverseFoldingStructureInput,
)
from bio_programming.tools.structures.structure import ProteinStructure

TEST_PDB_FILE = Path(__file__).parent.parent.parent / "dummy_data" / "renin_af3.pdb"


@pytest.fixture(scope="module")
def pdb_structure():
    return ProteinStructure(structure_filepath_or_content=TEST_PDB_FILE)


class TestProteinMPNNSample:

    @pytest.mark.uses_gpu
    def test_proteinmpnn_sample_simple(self, pdb_structure: ProteinStructure):
        input = InverseFoldingInput(
            inputs=[
                InverseFoldingStructureInput(structure=pdb_structure),
                InverseFoldingStructureInput(structure=pdb_structure),
                InverseFoldingStructureInput(structure=pdb_structure),
            ]
        )
        config = InverseFoldingConfig(
            batch_size=10, temperature=1.0, seed=42, keep_on_gpu=False
        )
        output = run_proteinmpnn_sample(input, config)
        assert (
            output.success
        ), f"Failed to sample protein sequences using ProteinMPNN: {output}"

        # Validate outputs
        for designed_sequences in output.designed_sequences:
            assert len(designed_sequences) == 10
            assert all(
                isinstance(sequence, str) for sequence in designed_sequences.sequences
            )
            assert all(
                isinstance(perplexity, float)
                for perplexity in designed_sequences.perplexity
            )
            assert all(
                isinstance(identity, float)
                for identity in designed_sequences.sequence_identity
            )

    @pytest.mark.uses_gpu
    def test_proteinmpnn_sample_advanced_args(self, pdb_structure: ProteinStructure):
        """
        A more complex tests that ensures that advanced config settings work as expected.
        Tests the following advanced config settings:
        - fixed_positions
        - excluded_amino_acids
        """
        chain_A = pdb_structure.get_chain_sequence("A")

        # Find all indicies of the amino acid "C" in chain A
        c_positions = [i + 1 for i, aa in enumerate(chain_A) if aa == "C"]

        # Make a list of fixed indices that do not contain the "C" positions
        fixed_positions = random.sample(
            list(set(np.arange(len(chain_A)) + 1) - set(c_positions)), 200
        )

        input = InverseFoldingInput(
            inputs=[
                InverseFoldingStructureInput(
                    structure=pdb_structure, fixed_positions={"A": fixed_positions}
                ),
                InverseFoldingStructureInput(
                    structure=pdb_structure, fixed_positions={"A": fixed_positions}
                ),
            ]
        )
        config = InverseFoldingConfig(
            batch_size=10,
            temperature=1.0,
            seed=42,
            excluded_amino_acids=["C"],
            keep_on_gpu=False,
        )

        output = run_proteinmpnn_sample(input, config)
        assert (
            output.success
        ), f"Failed to sample protein sequences using ProteinMPNN: {output}"

        # Validate outputs followed the advanced config settings
        for designed_sequences in output.designed_sequences:
            for sequence in designed_sequences.sequences:
                # Ensure that the sequence does not contain the "C" amino acid
                # - We hardcoded that all C positions are designable above
                # - We do not allow C amino acids in the sequence
                assert (
                    "C" not in sequence
                ), f"Sequence contains 'C' amino acid: {sequence}"

                # Ensure that the sequence does not contain the fixed indices
                for position in fixed_positions:
                    assert (
                        sequence[position - 1] == chain_A[position - 1]
                    ), f"Designed sequence is not the same as the PDB prompt sequence at fixed position {position} (index {position-1}): {sequence[position-1]} != {chain_A[position-1]}"


class TestProteinMPNNScore:

    @pytest.mark.uses_gpu
    def test_proteinmpnn_score(self, pdb_structure: ProteinStructure):
        original_sequence = pdb_structure.get_chain_sequence("A")

        # Randomly change a few positions to 'C'
        modified_sequence = list(original_sequence)
        for index in random.sample(range(len(modified_sequence)), 100):
            modified_sequence[index] = "C"
        modified_sequence = "".join(modified_sequence)

        # Randomly select 100 positions to fix
        fixed_positions = random.sample(list(range(len(original_sequence))), 100)
        fixed_positions = {
            "A": [position + 1 for position in fixed_positions],
        }

        input = ProteinMPNNScoringInput(
            sequence_structure_pairs=[
                SequenceStructurePair(
                    sequence=original_sequence, structure=pdb_structure
                ),
                SequenceStructurePair(
                    sequence=modified_sequence, structure=pdb_structure
                ),
            ]
        )
        config = ProteinMPNNScoringConfig(
            fixed_positions=fixed_positions, seed=42, keep_on_gpu=False
        )
        output = run_proteinmpnn_score(input, config)
        assert (
            output.success
        ), f"Failed to score protein sequence using ProteinMPNN: {output}"

        # Validate outputs
        assert len(output.scores) == 2
        assert all(isinstance(score, SequenceScores) for score in output.scores)

        # Ensure the perplexity of the original sequence is lower than the modified sequence
        assert output.scores[0].perplexity < output.scores[1].perplexity

    @pytest.mark.uses_gpu
    def test_proteinmpnn_score_fields(self, pdb_structure: ProteinStructure):
        """Test all scoring fields and their mathematical relationships."""
        original_sequence = pdb_structure.get_chain_sequence("A")
        seq_len = len(original_sequence)

        input = ProteinMPNNScoringInput(
            sequence_structure_pairs=[
                SequenceStructurePair(
                    sequence=original_sequence, structure=pdb_structure
                ),
            ]
        )
        config = ProteinMPNNScoringConfig(seed=42, keep_on_gpu=False)
        output = run_proteinmpnn_score(input, config)
        assert output.success

        score = output.scores[0]

        # Validate metrics via attribute access (via __getattr__)
        assert isinstance(score.log_likelihood, float)
        assert isinstance(score.avg_log_likelihood, float)
        assert isinstance(score.perplexity, float)

        # Validate metrics via dict access
        assert isinstance(score.metrics["log_likelihood"], float)
        assert isinstance(score.metrics["avg_log_likelihood"], float)
        assert isinstance(score.metrics["perplexity"], float)

        # Validate logits
        assert isinstance(score.logits, np.ndarray)

        # Validate mathematical relationships
        # log_likelihood = avg_log_likelihood * seq_len
        assert np.isclose(
            score.log_likelihood, score.avg_log_likelihood * seq_len, rtol=1e-5
        )

        # perplexity = exp(-avg_log_likelihood)
        assert np.isclose(
            score.perplexity, np.exp(-score.avg_log_likelihood), rtol=1e-5
        )

        # avg_log_likelihood should be negative (log probs are <= 0)
        assert score.avg_log_likelihood <= 0

        # perplexity should be >= 1 (since exp(-x) >= 1 when x <= 0)
        assert score.perplexity >= 1.0

        # Validate logits shape: (seq_len, alphabet_size)
        assert score.logits.shape == (seq_len, len(MPNN_ALPHABET))

    @pytest.mark.uses_gpu
    def test_proteinmpnn_score_alphabet(self, pdb_structure: ProteinStructure):
        """Test the alphabet property on output."""
        original_sequence = pdb_structure.get_chain_sequence("A")

        input = ProteinMPNNScoringInput(
            sequence_structure_pairs=[
                SequenceStructurePair(
                    sequence=original_sequence, structure=pdb_structure
                ),
            ]
        )
        config = ProteinMPNNScoringConfig(seed=42, keep_on_gpu=False)
        output = run_proteinmpnn_score(input, config)
        assert output.success

        # Test alphabet on output
        assert output.alphabet == MPNN_ALPHABET

    @pytest.mark.uses_gpu
    def test_proteinmpnn_score_export_json(self, pdb_structure: ProteinStructure):
        """Test JSON export of scoring output."""
        original_sequence = pdb_structure.get_chain_sequence("A")

        input = ProteinMPNNScoringInput(
            sequence_structure_pairs=[
                SequenceStructurePair(
                    sequence=original_sequence, structure=pdb_structure
                ),
            ]
        )
        config = ProteinMPNNScoringConfig(seed=42, keep_on_gpu=False)
        output = run_proteinmpnn_score(input, config)
        assert output.success

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test export to directory (auto-naming)
            output._export_output(tmpdir, "json")
            json_path = Path(tmpdir) / "scores.json"
            assert json_path.exists()

            with open(json_path) as f:
                data = json.load(f)

            assert len(data) == 1
            assert "log_likelihood" in data[0]
            assert "avg_log_likelihood" in data[0]
            assert "perplexity" in data[0]
            assert "logits" in data[0]

            # Verify values match (using metrics dict or attribute access)
            assert np.isclose(data[0]["perplexity"], output.scores[0].metrics["perplexity"])
            assert np.isclose(data[0]["log_likelihood"], output.scores[0].metrics["log_likelihood"])

    @pytest.mark.uses_gpu
    def test_proteinmpnn_score_export_csv(self, pdb_structure: ProteinStructure):
        """Test CSV export of scoring output."""
        import csv

        original_sequence = pdb_structure.get_chain_sequence("A")

        input = ProteinMPNNScoringInput(
            sequence_structure_pairs=[
                SequenceStructurePair(
                    sequence=original_sequence, structure=pdb_structure
                ),
                SequenceStructurePair(
                    sequence=original_sequence, structure=pdb_structure
                ),
            ]
        )
        config = ProteinMPNNScoringConfig(seed=42, keep_on_gpu=False)
        output = run_proteinmpnn_score(input, config)
        assert output.success

        with tempfile.TemporaryDirectory() as tmpdir:
            output._export_output(tmpdir, "csv")
            csv_path = Path(tmpdir) / "scores.csv"
            assert csv_path.exists()

            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            assert set(rows[0].keys()) == {"log_likelihood", "avg_log_likelihood", "perplexity"}

            # Verify values match (CSV stores as strings)
            assert np.isclose(float(rows[0]["perplexity"]), output.scores[0].perplexity)

    @pytest.mark.uses_gpu
    def test_proteinmpnn_score_cache(self, pdb_structure: ProteinStructure):
        """
        Tests the caching functionality of ProteinMPNN scoring tool
        """
        from bio_programming.tools.tool_cache import (
            ToolCache,
            _program_tool_cache,
            get_cache_info,
        )

        original_sequence = pdb_structure.get_chain_sequence("A")

        # Create three different modified sequences
        modified_sequence_1 = list(original_sequence)
        for index in random.sample(range(len(modified_sequence_1)), 50):
            modified_sequence_1[index] = "A"
        modified_sequence_1 = "".join(modified_sequence_1)

        modified_sequence_2 = list(original_sequence)
        for index in random.sample(range(len(modified_sequence_2)), 50):
            modified_sequence_2[index] = "G"
        modified_sequence_2 = "".join(modified_sequence_2)

        modified_sequence_3 = list(original_sequence)
        for index in random.sample(range(len(modified_sequence_3)), 50):
            modified_sequence_3[index] = "V"
        modified_sequence_3 = "".join(modified_sequence_3)

        # Set up the cache
        cache = ToolCache()
        _program_tool_cache.set(cache)

        try:
            # First pass: score original and first two modified sequences
            input_first_pass = ProteinMPNNScoringInput(
                sequence_structure_pairs=[
                    SequenceStructurePair(
                        sequence=original_sequence, structure=pdb_structure
                    ),
                    SequenceStructurePair(
                        sequence=modified_sequence_1, structure=pdb_structure
                    ),
                    SequenceStructurePair(
                        sequence=modified_sequence_2, structure=pdb_structure
                    ),
                ]
            )
            config = ProteinMPNNScoringConfig(seed=42, keep_on_gpu=False)
            output_first_pass = run_proteinmpnn_score(input_first_pass, config)

            # Verify first pass succeeded
            assert (
                output_first_pass.success
            ), f"Failed to score protein sequence using ProteinMPNN: {output_first_pass}"
            assert len(output_first_pass.scores) == 3

            # Cache should have three entries
            cache_info = get_cache_info()
            assert cache_info["total_entries"] == 3

            # Second pass: score with overlapping sequences plus one new sequence
            input_second_pass = ProteinMPNNScoringInput(
                sequence_structure_pairs=[
                    SequenceStructurePair(
                        sequence=original_sequence, structure=pdb_structure
                    ),
                    SequenceStructurePair(
                        sequence=modified_sequence_1, structure=pdb_structure
                    ),
                    SequenceStructurePair(
                        sequence=modified_sequence_2, structure=pdb_structure
                    ),
                    SequenceStructurePair(
                        sequence=modified_sequence_3, structure=pdb_structure
                    ),
                ]
            )
            output_second_pass = run_proteinmpnn_score(input_second_pass, config)

            # Verify second pass succeeded
            assert (
                output_second_pass.success
            ), f"Failed to score protein sequence using ProteinMPNN: {output_second_pass}"
            assert len(output_second_pass.scores) == 4

            # Cache should have four entries (first three were cached, one new)
            cache_info = get_cache_info()
            assert cache_info["total_entries"] == 4

            # Verify scores are consistent - first three should match exactly
            assert (
                output_second_pass.scores[0].perplexity
                == output_first_pass.scores[0].perplexity
            )
            assert (
                output_second_pass.scores[1].perplexity
                == output_first_pass.scores[1].perplexity
            )
            assert (
                output_second_pass.scores[2].perplexity
                == output_first_pass.scores[2].perplexity
            )

            # Verify logits arrays are identical for cached results
            assert np.allclose(
                output_second_pass.scores[0].logits, output_first_pass.scores[0].logits
            )
            assert np.allclose(
                output_second_pass.scores[1].logits, output_first_pass.scores[1].logits
            )
            assert np.allclose(
                output_second_pass.scores[2].logits, output_first_pass.scores[2].logits
            )

        finally:
            # Clean up cache
            _program_tool_cache.set(None)
