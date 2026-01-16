"""
Unit tests for self-play message sequence generation.

The key test is verifying that each player's conversation history is built
correctly with asymmetric observations:
- Each player sees their own full response
- Each player sees partner's response as "Partner: [public part]"
- Error messages only go to the erring player
"""

import sys
from pathlib import Path

# Add dialop to path
DIALOP_PATH = Path(__file__).parent.parent.parent.parent / "self_play" / "scripts" / "dialop"
sys.path.insert(0, str(DIALOP_PATH.parent))

import pytest
from dialop.envs.optimization import OptimizationEnv
from agent_system import get_player_conversations, merge_consecutive_user_messages


class TestAsymmetricObservations:
    """Test that players see asymmetric views of the conversation."""

    def setup_method(self):
        """Set up a fresh environment for each test."""
        self.env = OptimizationEnv(max_turns=10, max_retries_per_turn=3)

    def test_initial_observations_are_different(self):
        """Each player should get a different initial observation (different tables)."""
        obss = self.env.reset(seed=42)

        p1_initial = obss["player-1"]
        p2_initial = obss["player-2"]

        # Both should have content
        assert len(p1_initial) > 0
        assert len(p2_initial) > 0

        # They should be different (different table views)
        assert p1_initial != p2_initial

        # Both should contain table data
        assert "Ava Li" in p1_initial  # Reviewer name
        assert "Ava Li" in p2_initial

    def test_message_asymmetry(self):
        """When one player sends a message, they see full text but partner sees 'Partner: ...'"""
        conversations = get_player_conversations(
            self.env,
            game_state=None,
            responses=[
                ("player-1", "[message] I see high scores for BLEU paper."),
            ]
        )

        # Player 1 should have: initial obs, their response, next turn prompt
        p1_msgs = conversations["player-1"]
        assert len(p1_msgs) >= 2

        # Player 1's assistant message should be their full response
        p1_response = next(m for m in p1_msgs if m["role"] == "assistant")
        assert "[message]" in p1_response["content"]
        assert "I see high scores" in p1_response["content"]

        # Player 2 should see "Partner: ..." in their observation
        p2_msgs = conversations["player-2"]
        p2_user_msgs = [m for m in p2_msgs if m["role"] == "user"]

        # Find the message that contains partner's utterance
        partner_obs = None
        for m in p2_user_msgs:
            if "Partner:" in m["content"]:
                partner_obs = m["content"]
                break

        assert partner_obs is not None, "Player 2 should see Partner: prefix"
        assert "Partner:" in partner_obs
        assert "[message]" in partner_obs

    def test_error_only_visible_to_erring_player(self):
        """When a player makes an error, only they see the error message."""
        # Send a malformed message (no [message] tag)
        conversations = get_player_conversations(
            self.env,
            game_state=None,
            responses=[
                ("player-1", "This message has no tag and should error"),
            ]
        )

        p1_msgs = conversations["player-1"]
        p2_msgs = conversations["player-2"]

        # Player 1 should see an error message
        p1_user_msgs = [m["content"] for m in p1_msgs if m["role"] == "user"]
        has_error = any("Error" in msg or "error" in msg.lower() for msg in p1_user_msgs)
        assert has_error, "Player 1 should receive error feedback"

        # Player 2 should NOT see the error (only initial observation)
        # After an error, the other player's observation should be empty
        p2_user_msgs = [m["content"] for m in p2_msgs if m["role"] == "user"]
        # Player 2 should only have their initial observation
        assert len(p2_user_msgs) == 1, "Player 2 should only see initial obs, not the error"

    def test_proposal_visibility(self):
        """When one player proposes, both see it but partner gets accept/reject prompt."""
        # Create a valid proposal format
        proposal = """[propose] Here's my proposal.

Proposal:
- BLEU: Ava Li
- Electra: Daniel Nguyen
- GloVe: Sofia Patel
- GLUE: Andrei Petrov
- LLaMA: Morgan Reed
- RoBERTa: Joseph Santos
- QuAC: Ethan Smith
- SWAG: Noah Wilson"""

        conversations = get_player_conversations(
            self.env,
            game_state=None,
            responses=[
                ("player-1", proposal),
            ]
        )

        p1_msgs = conversations["player-1"]
        p2_msgs = conversations["player-2"]

        # Player 2 should see accept/reject prompt
        p2_content = " ".join(m["content"] for m in p2_msgs if m["role"] == "user")
        assert "[accept]" in p2_content or "accept" in p2_content.lower()
        assert "[reject]" in p2_content or "reject" in p2_content.lower()

    def test_multi_turn_conversation(self):
        """Test a multi-turn exchange builds correct histories."""
        conversations = get_player_conversations(
            self.env,
            game_state=None,
            responses=[
                ("player-1", "[message] I have high scores for BLEU and GloVe papers."),
                ("player-2", "[message] I see good matches for Electra and GLUE."),
                ("player-1", "[message] Let's try to find the optimal assignment."),
            ]
        )

        p1_msgs = conversations["player-1"]
        p2_msgs = conversations["player-2"]

        # Count assistant messages for each player
        p1_assistant_count = sum(1 for m in p1_msgs if m["role"] == "assistant")
        p2_assistant_count = sum(1 for m in p2_msgs if m["role"] == "assistant")

        # Player 1 made 2 moves, Player 2 made 1
        assert p1_assistant_count == 2, f"Expected 2 assistant msgs for P1, got {p1_assistant_count}"
        assert p2_assistant_count == 1, f"Expected 1 assistant msg for P2, got {p2_assistant_count}"

        # Both players should see each other's messages as "Partner: ..."
        p1_content = " ".join(m["content"] for m in p1_msgs)
        p2_content = " ".join(m["content"] for m in p2_msgs)

        # P1 should see P2's message as partner
        assert "Partner:" in p1_content and "Electra" in p1_content

        # P2 should see P1's messages as partner
        assert "Partner:" in p2_content and "BLEU" in p2_content


class TestGameStateReproducibility:
    """Test that games can be reproduced from saved state."""

    def test_seed_produces_same_game(self):
        """Same seed should produce identical initial observations."""
        env1 = OptimizationEnv(max_turns=10)
        env2 = OptimizationEnv(max_turns=10)

        obss1 = env1.reset(seed=12345)
        obss2 = env2.reset(seed=12345)

        assert obss1["player-1"] == obss2["player-1"]
        assert obss1["player-2"] == obss2["player-2"]

    def test_different_seeds_produce_different_games(self):
        """Different seeds should produce different games."""
        env1 = OptimizationEnv(max_turns=10)
        env2 = OptimizationEnv(max_turns=10)

        obss1 = env1.reset(seed=11111)
        obss2 = env2.reset(seed=22222)

        # Very likely to be different (not guaranteed but extremely probable)
        assert obss1["player-1"] != obss2["player-1"]


class TestConversationStructure:
    """Test the structure of generated conversations."""

    def setup_method(self):
        self.env = OptimizationEnv(max_turns=10, max_retries_per_turn=3)

    def test_player_does_not_see_own_message_echoed(self):
        """The sender does NOT see their own message echoed back.

        Correct behavior (matching generate_rollouts.py):
        1. Player sends response (assistant)
        2. Only the OTHER player observes "Partner: ..."
        3. Sender's own message is already in their history as assistant

        This avoids redundant consecutive user messages.
        """
        conversations = get_player_conversations(
            self.env,
            game_state=None,
            responses=[
                ("player-1", "[message] First message."),
                ("player-2", "[message] Second message."),
            ]
        )

        p1_msgs = conversations["player-1"]
        user_msgs = [m for m in p1_msgs if m["role"] == "user"]

        # Player 1's user messages should be: initial obs, then partner's response
        # NOT their own message echoed back
        echoed = [m for m in user_msgs if "First message" in m["content"] and "Partner:" not in m["content"]]
        assert len(echoed) == 0, "Player should NOT see their own message echoed in user messages"

        # They should see partner's message
        partner_msgs = [m for m in user_msgs if "Partner:" in m["content"]]
        assert len(partner_msgs) >= 1, "Player should see partner's message"

        # Their own message should be in assistant messages only
        assistant_msgs = [m for m in p1_msgs if m["role"] == "assistant"]
        own_msg = [m for m in assistant_msgs if "First message" in m["content"]]
        assert len(own_msg) == 1, "Player's own message should be in assistant history"

    def test_first_message_is_user(self):
        """Each player's conversation should start with a user message (initial obs)."""
        conversations = get_player_conversations(
            self.env,
            game_state=None,
            responses=[]
        )

        for player_key, msgs in conversations.items():
            assert len(msgs) > 0, f"{player_key} should have initial observation"
            assert msgs[0]["role"] == "user", \
                f"{player_key}'s first message should be user (initial obs)"

    def test_merge_user_messages_option(self):
        """Test that merge_user_messages=True produces alternating user/assistant."""
        conversations = get_player_conversations(
            self.env,
            game_state=None,
            responses=[
                ("player-1", "[message] First message."),
                ("player-2", "[message] Second message."),
            ],
            merge_user_messages=True
        )

        for player_key, msgs in conversations.items():
            prev_role = None
            for msg in msgs:
                if prev_role is not None:
                    assert msg["role"] != prev_role, \
                        f"{player_key}: consecutive {msg['role']} messages after merge"
                prev_role = msg["role"]


class TestMergeUtility:
    """Test the merge_consecutive_user_messages utility."""

    def test_merge_consecutive_users(self):
        """Consecutive user messages should be merged."""
        messages = [
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"},
            {"role": "assistant", "content": "Response"},
        ]

        merged = merge_consecutive_user_messages(messages)

        assert len(merged) == 2
        assert merged[0]["role"] == "user"
        assert "First" in merged[0]["content"]
        assert "Second" in merged[0]["content"]
        assert merged[1]["role"] == "assistant"

    def test_no_merge_when_alternating(self):
        """Already alternating messages should be unchanged."""
        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]

        merged = merge_consecutive_user_messages(messages)

        assert len(merged) == 3
        assert merged[0]["content"] == "Q1"
        assert merged[1]["content"] == "A1"
        assert merged[2]["content"] == "Q2"

    def test_empty_list(self):
        """Empty list should return empty."""
        assert merge_consecutive_user_messages([]) == []


def print_conversations(conversations: dict):
    """Helper to visualize conversation histories for debugging."""
    for player, msgs in conversations.items():
        print(f"\n{'='*60}")
        print(f"  {player.upper()}")
        print(f"{'='*60}")
        for i, msg in enumerate(msgs):
            role = msg["role"].upper()
            content = msg["content"]
            # Truncate long content
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"\n[{i}] {role}:")
            print(f"    {content}")


if __name__ == "__main__":
    # Run a quick visual test
    print("Running visual test of conversation generation...")

    env = OptimizationEnv(max_turns=10, max_retries_per_turn=3)

    # Test a simple exchange
    conversations = get_player_conversations(
        env,
        game_state=None,
        responses=[
            ("player-1", "[message] I see some high scores in my table."),
            ("player-2", "[message] Me too! Let me share what I know."),
        ]
    )

    print_conversations(conversations)

    # Test an error case
    print("\n\n" + "="*60)
    print("Testing error case (malformed message):")
    print("="*60)

    env2 = OptimizationEnv(max_turns=10, max_retries_per_turn=3)
    conversations_error = get_player_conversations(
        env2,
        game_state=None,
        responses=[
            ("player-1", "No tag here, should error"),
        ]
    )

    print_conversations(conversations_error)

    print("\n\nRunning pytest...")
    pytest.main([__file__, "-v"])
