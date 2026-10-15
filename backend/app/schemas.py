"""Pydantic request/response schemas (input validation lives here)."""
import re

from pydantic import BaseModel, EmailStr, Field, field_validator

from .domain import COLOUR_SCHEMES, ROOM_TYPES, STYLES


class RegisterIn(BaseModel):
    name: str = Field(min_length=2, max_length=60)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    consent: bool

    @field_validator("name")
    @classmethod
    def clean_name(cls, v: str) -> str:
        v = v.strip()
        if not re.fullmatch(r"[\w .'\-]+", v):
            raise ValueError("Name contains invalid characters")
        return v

    @field_validator("password")
    @classmethod
    def strong_password(cls, v: str) -> str:
        if not (re.search(r"[A-Za-z]", v) and re.search(r"\d", v)):
            raise ValueError("Password must contain at least one letter and one number")
        return v

    @field_validator("consent")
    @classmethod
    def must_consent(cls, v: bool) -> bool:
        if not v:
            raise ValueError("You must accept the privacy notice to create an account")
        return v


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class TokenOut(BaseModel):
    token: str
    user: dict


class PreferencesIn(BaseModel):
    room_type: str = "auto"
    style: str
    scheme: str
    length_m: float = Field(ge=1.5, le=20)
    width_m: float = Field(ge=1.5, le=20)
    height_m: float = Field(default=2.4, ge=2.0, le=6)
    budget: float = Field(ge=50, le=100_000)
    keep_existing: bool = False

    @field_validator("room_type")
    @classmethod
    def valid_room(cls, v):
        if v != "auto" and v not in ROOM_TYPES:
            raise ValueError(f"room_type must be 'auto' or one of {list(ROOM_TYPES)}")
        return v

    @field_validator("style")
    @classmethod
    def valid_style(cls, v):
        if v not in STYLES:
            raise ValueError(f"style must be one of {list(STYLES)}")
        return v

    @field_validator("scheme")
    @classmethod
    def valid_scheme(cls, v):
        if v not in COLOUR_SCHEMES:
            raise ValueError(f"scheme must be one of {list(COLOUR_SCHEMES)}")
        return v


class DesignUpdateIn(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=80)
    favourite: bool | None = None
    selected_option: int | None = Field(default=None, ge=0, le=2)
    preferences: PreferencesIn | None = None


class RenderIn(BaseModel):
    option_index: int = Field(default=0, ge=0, le=2)


class PreviewIn(BaseModel):
    preferences: PreferencesIn


class SwapIn(BaseModel):
    category: str = Field(min_length=1, max_length=40)
    sku: str = Field(min_length=1, max_length=40)


class EditIn(BaseModel):
    region: str = Field(pattern=r"^[a-z0-9-]{1,40}$")
    colour: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")


class EditsIn(BaseModel):
    edits: list[EditIn] = Field(default_factory=list, max_length=30)


class AIEditIn(BaseModel):
    action: str = Field(pattern=r"^(replace|remove)$")
    region: str = Field(pattern=r"^[a-z0-9-]{1,40}$")
    # Plain descriptive text only (it becomes part of an image prompt).
    prompt: str | None = Field(default=None, min_length=3, max_length=120, pattern=r"^[A-Za-z0-9 ,.'&/()-]+$")


class GoogleIn(BaseModel):
    credential: str = Field(min_length=20, max_length=4096)


class ProfileIn(BaseModel):
    name: str = Field(min_length=2, max_length=60)

    @field_validator("name")
    @classmethod
    def clean_name(cls, v: str) -> str:
        return RegisterIn.clean_name(v)


class PasswordIn(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)

    @field_validator("new_password")
    @classmethod
    def strong_password(cls, v: str) -> str:
        return RegisterIn.strong_password(v)
