"""
tests/unit/test_auth.py — Auth unit tests.
Password hashing, JWT creation/decode, exception handling.
"""

import pytest
from backend.security import hash_password, verify_password, create_access_token, decode_token
from backend.exceptions import TokenInvalid


class TestPasswordHashing:
    def test_hash_differs_from_plain(self):
        h = hash_password("mypassword")
        assert h != "mypassword"

    def test_verify_correct_password(self):
        h = hash_password("secret123")
        assert verify_password("secret123", h)

    def test_verify_wrong_password_fails(self):
        h = hash_password("secret123")
        assert not verify_password("wrong", h)

    def test_same_password_gives_different_hashes(self):
        # bcrypt uses random salt each time
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2


class TestJWT:
    def test_token_contains_correct_claims(self):
        token   = create_access_token("uid-1", "a@b.com", "free")
        payload = decode_token(token)
        assert payload["sub"]   == "uid-1"
        assert payload["email"] == "a@b.com"
        assert payload["tier"]  == "free"
        assert "exp" in payload
        assert "jti" in payload

    def test_invalid_token_raises_token_invalid(self):
        with pytest.raises(TokenInvalid):
            decode_token("not.a.valid.token")

    def test_tampered_token_raises(self):
        token   = create_access_token("uid-1", "a@b.com", "free")
        tampered = token[:-5] + "XXXXX"
        with pytest.raises(TokenInvalid):
            decode_token(tampered)

    def test_pro_tier_stored_correctly(self):
        token   = create_access_token("uid-2", "pro@b.com", "pro")
        payload = decode_token(token)
        assert payload["tier"] == "pro"