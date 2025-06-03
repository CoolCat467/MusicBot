"""Music streaming discord bot.

Original Source: https://gist.github.com/vbe0201/ade9b80f2d3b64643d854938d40a0a2d
"""

# Programmed by CoolCat467

from __future__ import annotations

__title__ = "MusicBot"
__author__ = "vbe0201, PiBoy, and CoolCat467"
__version__ = "1.0.0"


import asyncio
import difflib
import inspect
import io
import os
import sys
import traceback
from typing import TYPE_CHECKING, Any, Final, cast, get_args, get_type_hints

import discord
import yt_dlp
from discord.ext.commands import CommandError
from dotenv import load_dotenv

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine, Iterable

# https://discordpy.readthedocs.io/en/latest/index.html
# https://discord.com/developers


load_dotenv()
TOKEN: Final = os.getenv("DISCORD_TOKEN")

BOT_PREFIX: Final = "!music"
BOT_DESC: Final = "Play music by stealing it from youtube"


# Suppress noise about console usage from errors
yt_dlp.utils.bug_reports_message = lambda: ""


ytdl_format_options: Final = {
    "format": "bestaudio/best",
    "outtmpl": "%(extractor)s-%(id)s-%(title)s.%(ext)s",
    "restrictfilenames": True,
    "noplaylist": True,
    "nocheckcertificate": True,
    "ignoreerrors": False,
    "logtostderr": False,
    "quiet": True,
    "no_warnings": True,
    "default_search": "auto",
    "source_address": "0.0.0.0",  # noqa: S104
    # bind to ipv4 since ipv6 addresses cause issues sometimes
}

FFMPEG_OPTIONS: dict[str, Any] = {
    "before_options": "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5",
    "options": "-vn",
}

ytdl: Final = yt_dlp.YoutubeDL(ytdl_format_options)


def combine_end(data: Iterable[str], final: str = "and") -> str:
    """Join values of text, and have final with the last one properly."""
    data = list(map(str, data))
    if len(data) >= 2:
        data[-1] = f"{final} {data[-1]}"
    if len(data) > 2:
        return ", ".join(data)
    return " ".join(data)


def parse_args(string: str, ignore: int = 0, sep: str = " ") -> list[str]:
    """Return a list of arguments."""
    return string.split(sep)[ignore:]


def closest(given: str, options: Iterable[str]) -> str:
    """Get closest text to given from options."""
    return difflib.get_close_matches(given, options, n=1, cutoff=0)[0]


def log_active_exception(extra: str | None = None) -> str:
    """Log active exception."""
    # Get values from exc_info
    values = sys.exc_info()
    # Get error message.
    msg = "#" * 16 + "\n"
    if extra is not None:
        msg += f"{extra}\n"
    msg += "Exception class:\n" + str(values[0]) + "\n"
    msg += "Exception text:\n" + str(values[1]) + "\n"

    with io.StringIO() as yes_totaly_a_file:
        traceback.print_exception(
            None,
            value=values[1],
            tb=values[2],
            limit=None,
            file=yes_totaly_a_file,
            chain=True,
        )
        msg += "\n" + yes_totaly_a_file.getvalue() + "\n" + "#" * 16 + "\n"
    return msg


def union_match(argument_type: type, target_type: type) -> bool:
    """Return if argument type or optional of argument type is target type."""
    return argument_type == target_type or argument_type in get_args(
        target_type,
    )


def process_arguments(
    parameters: dict[str, type],
    given_args: list[str],
    message: discord.message.Message,
) -> dict[str, Any]:
    """Process arguments to on_message handler."""
    complete: dict[str, Any] = {}
    if not parameters:
        return complete

    required_count = 0
    for v in parameters.values():
        if type(None) not in get_args(v):
            required_count += 1

    if len(given_args) < required_count:
        raise ValueError("Missing parameters!")

    i = -1
    for name, target_type in parameters.items():
        i += 1
        arg = None if i >= len(given_args) else given_args[i]
        arg_type = type(arg)
        if union_match(arg_type, target_type):
            complete[name] = arg
            continue
        if union_match(arg_type, str):
            assert isinstance(arg, str)
            matched = False
            if message.guild is not None:
                if union_match(target_type, discord.VoiceChannel):
                    for voice_channel in message.guild.voice_channels:
                        if voice_channel.name == arg:
                            complete[name] = voice_channel
                            matched = True
                            break
                if union_match(target_type, discord.TextChannel):
                    for text_channel in message.guild.text_channels:
                        if text_channel.name == arg:
                            complete[name] = text_channel
                            matched = True
                            break
            if union_match(target_type, float) and arg.isdecimal():
                complete[name] = float(arg)
                continue
            if union_match(target_type, int) and arg.isdigit():
                complete[name] = int(arg)
                continue
            if matched:
                continue
        raise ValueError
    if parameters and union_match(target_type, str) and i < len(given_args):
        complete[name] += " " + " ".join(given_args[i:])
    return complete


def override_methods(obj: Any, attrs: dict[str, Any]) -> Any:
    """Override attributes of object."""

    class OverrideGetattr:
        """Override get attribute."""

        def __getattr__(self, attr_name: str, /, default: Any = None) -> Any:
            """Get attribute but maybe return proxy of attribute."""
            if attr_name not in attrs:
                if default is None:
                    return getattr(obj, attr_name)
                return getattr(obj, attr_name, default)
            return attrs[attr_name]

        # def __setattr__(self, attr_name: str, value: Any) -> None:
        #     setattr(obj, attr_name, value)
        def __repr__(self) -> str:
            return f"Overwritten {obj!r}"

    override = OverrideGetattr()
    for attr in dir(obj):
        if attr not in attrs and not attr.endswith("__"):
            try:
                setattr(override, attr, getattr(obj, attr))
            except AttributeError:
                print(attr)
    set_function_name = "__setattr__"
    setattr(
        override,
        set_function_name,
        lambda attr_name, value: setattr(obj, attr_name, value),
    )
    return override


def interaction_to_message(
    interaction: discord.Interaction[MusicBot],
) -> discord.Message:
    """Convert slash command interaction to Message."""

    def str_null(x: object | None) -> str | None:
        return None if x is None else str(x)

    data: dict[str, Any] = {
        "id": interaction.id,
        "webhook_id": None,
        "reactions": [],
        "attachments": [],
        "activity": None,
        "embeds": [],
        "edited_timestamp": None,
        "type": 0,  # discord.MessageType.default,
        "pinned": False,
        "flags": 0,
        "mention_everyone": False,
        "tts": False,
        "content": "",
        "nonce": None,  # Optional[Union[int, str]]
        "sticker_items": [],
        "guild_id": interaction.guild_id,
        "interaction": {
            "id": interaction.id,
            "type": 2,
            "name": "Interaction name",
            "member": {
                "joined_at": (
                    str_null(interaction.user.joined_at)
                    if isinstance(interaction.user, discord.Member)
                    else None
                ),
                "premium_since": (
                    str_null(interaction.user.premium_since)
                    if isinstance(interaction.user, discord.Member)
                    else None
                ),
                "roles": (
                    []
                    if isinstance(interaction.user, discord.User)
                    else [role.id for role in interaction.user.roles]
                ),
                "nick": (
                    interaction.user.nick
                    if isinstance(interaction.user, discord.Member)
                    else None
                ),
                "pending": (
                    interaction.user.pending
                    if isinstance(interaction.user, discord.Member)
                    else None
                ),
                "avatar": interaction.user.avatar,
                "flags": (
                    interaction.user._flags
                    if isinstance(interaction.user, discord.Member)
                    else None
                ),
                "permissions": (
                    interaction.user._permissions
                    if isinstance(interaction.user, discord.Member)
                    else None
                ),
                "communication_disabled_until": (
                    str_null(
                        interaction.user.timed_out_until,
                    )
                    if isinstance(interaction.user, discord.Member)
                    else None
                ),
            },
            "user": {
                "username": interaction.user.name,
                "id": interaction.user.id,
                "discriminator": interaction.user.discriminator,
                "avatar": interaction.user._avatar,
                "bot": interaction.user.bot,
                "system": interaction.user.system,
                "roles": (
                    []
                    if isinstance(interaction.user, discord.User)
                    else [role.id for role in interaction.user.roles]
                ),
            },
        },
        # 'message_reference': None,
        "application": {
            "id": interaction.application_id,
            "description": "Application description",
            "name": "Application name",
            "icon": None,
            "cover_image": None,
        },
        # 'author'       : ,
        # 'member'       : ,
        # 'mentions'     : ,
        # 'mention_roles': ,
        # 'components'   :
    }

    message = discord.message.Message(
        state=interaction._state,
        channel=interaction.channel,  # type: ignore
        data=data,  # type: ignore
    )

    message.author = interaction.user

    channel_send = message.channel.send
    times = -1

    async def send_message(*args: Any, **kwargs: Any) -> Any:
        """Send message."""
        nonlocal times
        times += 1
        if times == 0:
            return await interaction.response.send_message(*args, **kwargs)
        return await channel_send(*args, **kwargs)

    message.channel = override_methods(
        message.channel,
        {
            "send": send_message,
        },
    )

    return message


def extract_parameters_from_callback(
    func: Callable[..., Any],
    globalns: dict[str, Any],
) -> dict[str, discord.app_commands.transformers.CommandParameter]:
    """Set up slash command things from function.

    Stolen from internals of discord.app_commands.commands
    """
    params = inspect.signature(func).parameters
    cache: dict[str, Any] = {}
    required_params = 1
    if len(params) < required_params:
        raise TypeError(
            f"callback {func.__qualname__!r} must have more "
            f"than {required_params - 1} parameter(s)",
        )

    iterator = iter(params.values())
    for _ in range(required_params):
        next(iterator)

    parameters: list[discord.app_commands.transformers.CommandParameter] = []
    for parameter in iterator:
        if parameter.annotation is parameter.empty:
            raise TypeError(
                f"parameter {parameter.name!r} is missing a "
                f"type annotation in callback {func.__qualname__!r}",
            )

        resolved = discord.utils.resolve_annotation(
            parameter.annotation,
            globalns,
            globalns,
            cache,
        )
        param = discord.app_commands.transformers.annotation_to_parameter(
            resolved,
            parameter,
        )
        parameters.append(param)

    values = sorted(parameters, key=lambda a: a.required, reverse=True)
    result = {v.name: v for v in values}

    descriptions = discord.app_commands.commands._parse_args_from_docstring(
        func,
        result,
    )

    try:
        name = "__discord_app_commands_param_description__"
        descriptions.update(getattr(func, name))
    except AttributeError:
        for param in values:
            if param.description is discord.utils.MISSING:
                param.description = "…"
    if descriptions:
        discord.app_commands.commands._populate_descriptions(
            result,
            descriptions,
        )

    try:
        renames = func.__discord_app_commands_param_rename__  # type: ignore
    except AttributeError:
        pass
    else:
        discord.app_commands.commands._populate_renames(result, renames.copy())

    try:
        choices = func.__discord_app_commands_param_choices__  # type: ignore
    except AttributeError:
        pass
    else:
        discord.app_commands.commands._populate_choices(result, choices.copy())

    try:
        name = "__discord_app_commands_param_autocomplete__"
        autocomplete = getattr(func, name)
    except AttributeError:
        pass
    else:
        discord.app_commands.commands._populate_autocomplete(
            result,
            autocomplete.copy(),
        )

    return result


def slash_handle(
    message_command: Callable[[discord.Message], Awaitable[None]],
) -> tuple[
    Callable[[discord.Interaction[MusicBot]], Coroutine[Any, Any, Any]],
    Any,
]:
    """Slash handle wrapper to convert interaction to message."""

    class Dummy:
        """Dummy class so required_params = 2 for slash_handler."""

        async def slash_handler(
            *args: discord.Interaction[MusicBot],
            **kwargs: Any,
        ) -> None:
            """Slash command wrapper for message-based command."""
            interaction: discord.Interaction[MusicBot] = args[1]
            try:
                msg = interaction_to_message(interaction)
            except Exception:
                root = os.path.split(os.path.abspath(__file__))[0]
                logpath = os.path.join(root, "log.txt")
                log_active_exception(logpath)
                raise
            try:
                await message_command(msg, *args[2:], **kwargs)
            except Exception:
                await msg.channel.send(
                    "An error occurred processing the slash command",
                )
                if hasattr(interaction._client, "on_error"):
                    await interaction._client.on_error(
                        "slash_command",
                        message_command.__name__,
                    )
                raise

    params = extract_parameters_from_callback(
        message_command,
        message_command.__globals__,
    )
    merp = Dummy()
    return merp.slash_handler, params  # type: ignore


async def send_over_2000(
    send_func: Callable[[str], Awaitable[None]],
    text: str,
    header: str = "",
    wrap_with: str = "",
    start: str = "",
) -> None:
    """Use send_func to send text in segments if required."""
    parts = [start + wrap_with + header]
    send = str(text)
    wrap_alloc = len(wrap_with)
    while send:
        cur_block = len(parts[-1])
        if cur_block < 2000:
            end = 2000 - (cur_block + wrap_alloc)
            add = send[0:end]
            send = send[end:]
            parts[-1] += add + wrap_with
        else:
            parts.append(wrap_with + header)

    # pylint: disable=wrong-spelling-in-comment
    # This would be great for asyncio.gather, but
    # I'm pretty sure that will throw off call order,
    # and it's quite important that everything stays in order.
    # coros = [send_func(part) for part in parts]
    # await asyncio.gather(*coros)
    for part in parts:
        await send_func(part)


async def send_command_list(
    commands: dict[
        str,
        Callable[[discord.message.Message], Coroutine[None, None, None]],
    ],
    name: str,
    channel: discord.abc.Messageable,
) -> None:
    """Send message on channel telling user about all valid name commands."""
    sort = sorted(commands.keys(), reverse=True)
    command_data = [f"`{v}` - {commands[v].__doc__}" for v in sort]
    await send_over_2000(
        channel.send,  # type: ignore
        "\n".join(command_data),
        start=f"{__title__}'s Valid {name} Commands:\n",
    )


class YTDLSource(discord.PCMVolumeTransformer[discord.FFmpegPCMAudio]):
    """YouTube DL source."""

    __slots__ = ("data", "title", "url")

    def __init__(
        self,
        source: discord.FFmpegPCMAudio,
        *,
        data: dict[str, str],
        volume: float = 0.5,
    ) -> None:
        """Initialize Youtube Audio Source."""
        super().__init__(source, volume)

        self.data = data

        self.title = data.get("title")
        self.url = data.get("url")

    @classmethod
    async def from_url(
        cls,
        url: str,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        stream: bool = False,
    ) -> YTDLSource:
        """Make YTDLSource from url/query."""
        loop = loop or asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None,
            lambda: cast(
                "dict[str, Any]",
                ytdl.extract_info(url, download=not stream),
            ),
        )

        if "entries" in data:
            # take first item from a playlist
            entries = data["entries"]
            if entries:
                data = entries[0]

        filename = data["url"] if stream else ytdl.prepare_filename(data)
        return cls(
            discord.FFmpegPCMAudio(filename, **FFMPEG_OPTIONS),
            data=data,
        )


class MusicBot(discord.Client):
    """Music Bot."""

    __slots__ = ("commands", "guild_voices", "guild_volumes")
    prefix: str = BOT_PREFIX

    def __init__(
        self,
        *,
        intents: discord.Intents,
        loop: asyncio.AbstractEventLoop,
        **options: Any,
    ) -> None:
        """Initialize MusicBot."""
        super().__init__(intents=intents, loop=loop, **options)

        self.commands: dict[
            str,
            Callable[[discord.message.Message], Awaitable[Any]],
        ] = {
            "join": self.join,
            "stream": self.stream,
            "volume": self.volume,
            "stop": self.stop,
            "help": self.help,
            "toggle": self.toggle,
        }

        self.tree = discord.app_commands.CommandTree(self)
        for command_name, command_function in self.commands.items():
            callback, params = slash_handle(command_function)
            command: discord.app_commands.commands.Command[
                Any,
                Any,
                None,
            ] = discord.app_commands.commands.Command(
                name=command_name,
                description=command_function.__doc__ or "",
                callback=callback,  # type: ignore [arg-type]
                nsfw=False,
                auto_locale_strings=True,
            )
            command._params = params
            command.checks = getattr(
                callback,
                "__discord_app_commands_checks__",
                [],
            )
            command._guild_ids = getattr(
                callback,
                "__discord_app_commands_default_guilds__",
                None,
            )
            command.default_permissions = getattr(
                callback,
                "__discord_app_commands_default_permissions__",
                None,
            )
            command.guild_only = getattr(
                callback,
                "__discord_app_commands_guild_only__",
                False,
            )
            command.binding = getattr(command_function, "__self__", None)
            self.tree.add_command(command)
        self.tree.on_error = self.on_error  # type: ignore[assignment]

        self.guild_voices: dict[int, discord.VoiceClient] = {}
        self.guild_volumes: dict[int, int] = {}

    async def register_commands(self, guild: discord.Guild) -> None:
        """Register commands for guild."""
        self.tree.copy_global_to(guild=guild)

        await self.tree.sync(guild=guild)

    async def eval_guild(self, guild: discord.Guild) -> int:
        """Evaluate guild. Return guild id."""
        guild_id = guild.id
        voice_state = guild.voice_client
        if voice_state is not None:
            await voice_state.disconnect(force=True)
        return guild_id

    async def eval_guilds(self) -> list[int]:
        """Return list of guild ids evaluated."""
        ids = []
        register = []
        for guild in self.guilds:
            register.append(self.register_commands(guild))
            ids.append(self.eval_guild(guild))
        await asyncio.gather(*register)
        return await asyncio.gather(*ids)

    async def on_ready(self) -> None:
        """Set up slash commands and change presence."""
        print(f"{self.user} has connected to Discord!")
        print(f"Prefix  : {self.prefix}")
        print(f"Intents : {self.intents}")

        print(f"\n{self.user} is connected to the following guilds:\n")
        guildnames = []
        for guild in self.guilds:
            guildnames.append(f"{guild.name} (id: {guild.id})")
        spaces = max(len(name) for name in guildnames)
        print(
            "\n" + "\n".join(name.rjust(spaces) for name in guildnames) + "\n",
        )

        ids = await self.eval_guilds()

        print("Guilds evaluated:\n" + "\n".join([str(x) for x in ids]) + "\n")

        act = discord.Activity(
            type=discord.ActivityType.watching,
            name=f"for {self.prefix}",
        )
        await self.change_presence(status=discord.Status.online, activity=act)

    async def help(
        self,
        message: discord.message.Message,
    ) -> None:
        """Get all valid options for guilds."""
        await send_command_list(self.commands, "Guild", message.channel)

    def update_volume(self, guild_id: int) -> None:
        """Update volume for guild voice client."""
        if guild_id not in self.guild_voices:
            return
        voice_client = self.guild_voices[guild_id]

        if guild_id not in self.guild_volumes:
            # TODO: Make database remembering volumes
            self.guild_volumes[guild_id] = 100

        if voice_client.source is not None and hasattr(
            voice_client.source,
            "volume",
        ):
            volume = self.guild_volumes[guild_id]
            voice_client.source.volume = volume / 100

    def set_volume(self, guild_id: int, volume: int) -> None:
        """Set volume for guild."""
        self.guild_volumes[guild_id] = volume

        self.update_volume(guild_id)

    async def connect_voice(
        self,
        guild_id: int,
        channel: discord.VoiceChannel,
    ) -> None:
        """Connect to a voice channel."""
        if guild_id in self.guild_voices:
            voice = self.guild_voices[guild_id]
            if voice.channel.id != channel.id and voice.is_connected():
                await voice.disconnect(force=True)

        voice = await channel.connect(self_deaf=True)
        self.guild_voices[guild_id] = voice

    #    @play.before_invoke
    #    @yt.before_invoke
    #    @stream.before_invoke
    async def ensure_voice(
        self,
        message: discord.message.Message,
    ) -> discord.VoiceClient:
        """Ensure connected to voice channel. Return VoiceClient for guild."""
        if message.guild is None:
            await message.channel.send("Error: Not in a guild")
            raise CommandError("Author's message not in a guild")
        if message.guild.id not in self.guild_voices:
            assert isinstance(message.author, discord.Member)
            if message.author.voice:
                channel = message.author.voice.channel
                if not isinstance(channel, discord.VoiceChannel):
                    await message.channel.send(
                        "You are not connected to a voice channel.",
                    )
                    raise CommandError(
                        "Author not connected to a voice channel.",
                    )
                await self.connect_voice(message.guild.id, channel)
                value = channel.mention or f"`{channel.name}`"
                await message.channel.send(f"Connected to {value}")
            else:
                await message.channel.send(
                    "You are not connected to a voice channel.",
                )
                raise CommandError("Author not connected to a voice channel.")
        voice_client = self.guild_voices[message.guild.id]
        if voice_client.is_playing() or voice_client.is_paused():
            voice_client.stop()
        if not voice_client.is_connected():
            del self.guild_voices[message.guild.id]
            return await self.ensure_voice(message)
        return voice_client

    @discord.app_commands.describe(channel="Voice Channel to connect to")
    async def join(
        self,
        message: discord.message.Message,
        channel: discord.VoiceChannel,
    ) -> None:
        """Connect to a voice channel."""
        if message.guild is None:
            await message.channel.send("Error: Not in a guild")
            raise CommandError("Author's message not in a guild")
        await self.connect_voice(message.guild.id, channel)

        await message.channel.send(f"Connected to {channel.mention}")

    #    async def play(
    #        self, message: discord.message.Message, query: str
    #    ) -> None:
    #        "Play audio from filesystem given query"
    #        voice_client = await self.ensure_voice(message)
    #        assert message.guild is not None
    #
    #        source = discord.PCMVolumeTransformer(
    #            discord.FFmpegPCMAudio(query)
    #        )
    #
    #        def after(exception: Exception | None = None) -> None:
    #            if exception is None:
    #                return
    #            print(f'Exception playing music: {exception!r}')
    #        voice_client.play(source, after = after)
    #
    #        if voice_client.source is None:
    #            voice_client.source = player
    #        self.update_volume(message.guild.id)
    #
    #        await message.channel.send(f'Now playing: `{query}`')

    #    async def yt(
    #        self, message: discord.message.Message, url: str
    #    ) -> None:
    #        "Play audio from a url (almost anything youtube_dl supports)"
    #        voice_client = await self.ensure_voice(message)
    #        assert message.guild is not None
    #
    #        def after(exception: Exception | None = None) -> None:
    #            if exception is None:
    #                return
    #            print(f'Exception playing music: {exception!r}')
    #
    #        async with message.channel.typing():
    #            player = await YTDLSource.from_url(url, loop=self.loop)
    #            voice_client.play(player, after = after)
    #
    #        if voice_client.source is None:
    #            voice_client.source = player
    #        self.update_volume(message.guild.id)
    #
    #        await message.channel.send(f'Now playing: `{player.title}`')

    @discord.app_commands.rename(url="search")
    @discord.app_commands.describe(url="URL or query for youtube video")
    async def stream(self, message: discord.message.Message, url: str) -> None:
        """Stream audio from a youtube video from given URL."""
        voice_client = await self.ensure_voice(message)
        assert message.guild is not None

        def after(exception: Exception | None = None) -> None:
            if exception is None:
                return
            print(f"Exception playing music: {exception!r}")

        async with message.channel.typing():
            await message.channel.send("Searching for video...")
            player = await YTDLSource.from_url(
                url,
                loop=self.loop,
                stream=True,
            )
            voice_client.play(player, after=after)

            if voice_client.source is None:
                voice_client.source = player
            self.update_volume(message.guild.id)

            await message.channel.send(
                f"Now playing: `{player.title}`",
            )  # from {player.url}

    @discord.app_commands.describe(volume="Volume percentage")
    async def volume(
        self,
        message: discord.message.Message,
        volume: int,
    ) -> None:
        """Change the audio player's volume."""
        if message.guild is None:
            await message.channel.send("Error: Not in a guild")
            raise CommandError("Author's message not in a guild")

        if message.guild.id not in self.guild_voices:
            await message.channel.send("Not connected to a voice channel.")
            return

        self.set_volume(message.guild.id, volume)

        await message.channel.send(f"Changed volume to {volume}%")

    async def toggle(self, message: discord.message.Message) -> None:
        """Toggle playback of the current audio if there is anything playing."""
        if message.guild is None:
            await message.channel.send("Error: Not in a guild")
            raise CommandError("Author's message not in a guild")

        if message.guild.id not in self.guild_voices:
            await message.channel.send("No audio currently playing.")
            return
        voice_client = self.guild_voices[message.guild.id]

        if voice_client.is_playing():
            voice_client.pause()
            await message.channel.send("Paused audio")
        elif voice_client.is_paused():
            voice_client.resume()
            await message.channel.send("Resumed audio")

    async def stop(self, message: discord.message.Message) -> None:
        """Disconnects the bot from voice channel."""
        if message.guild is None:
            await message.channel.send("Error: Not in a guild")
            raise CommandError("Author's message not in a guild")

        guild_id = message.guild.id

        if guild_id not in self.guild_voices:
            await message.channel.send("Not connected to a voice channel.")
            return
        voice_client = self.guild_voices[guild_id]

        await voice_client.disconnect(force=True)
        del self.guild_voices[guild_id]
        await message.channel.send("Stopped audio")

    async def process_command_message(
        self,
        message: discord.message.Message,
    ) -> None:
        """Process new command message. Calls self.command[command](message)."""
        err = (
            " Please enter a valid command. Use `help` to see valid commands."
        )
        # Get content of message.
        content = message.content

        # If no space in message
        if " " not in content:
            if content == self.prefix:
                await message.channel.send("No command given." + err)
            return
        args = parse_args(content)

        # Get command. zeroth if direct, first if guild because of prefix.
        command = args[1].lower()

        if command not in self.commands:
            # Otherwise, send error of no command.
            best = closest(command, tuple(self.commands))
            suggest = f"Did you mean `{best}`?"
            await message.channel.send(
                f"No valid command given. {suggest}{err}",
            )
            return

        command_func = self.commands[command]

        annotations = get_type_hints(command_func)
        params = {}
        for name, typeval in annotations.items():
            if name in {"return"}:
                continue
            if typeval in {discord.Message}:
                continue

            params[name] = typeval

        try:
            command_args = process_arguments(params, args[2:], message)
        except ValueError:
            print(log_active_exception())
            names = combine_end(
                [
                    (
                        f"{k}"
                        if not isinstance(v, type)
                        else f"{k} ({v.__name__})"
                    )
                    for k, v in params.items()
                ],
            )
            await message.channel.send(
                f"Missing one or more arguments: {names}",
            )
            return

        # If command is valid, run it.
        await command_func(message, **command_args)

    # Intents.dm_messages, Intents.guild_messages, Intents.messages
    async def on_message(self, message: discord.message.Message) -> None:
        """React to any new messages."""
        # Skip messages from ourselves.
        if message.author == self.user:
            return

        # If we can send message to person,
        if not hasattr(message.channel, "send"):
            return

        # If message is from a guild,
        if not isinstance(message.guild, discord.guild.Guild):
            return

        # If message starts with our prefix,
        args = parse_args(message.clean_content.lower())
        pfx = args[0] == self.prefix if len(args) >= 1 else False
        # of it starts with us being mentioned,
        meant = False
        if message.content.startswith("<@"):
            new = message.content.replace("!", "")
            new = new.replace("&", "")
            assert self.user is not None, "self.user is None"
            meant = new.startswith(self.user.mention)
        if pfx or meant:
            # we are, in reality, the fastest typer in world. aw yep.
            async with message.channel.typing():
                # Process message as guild
                await self.process_command_message(message)

    # Default, not affected by intents
    async def on_error(
        self,
        event: str,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> None:  # pylint: disable=arguments-differ
        """Log error and continue."""
        if event == "on_message":
            print(f"Unhandled message: {args[0]}")
        extra = "Error Event:\n" + str(event) + "\n"
        extra += (
            "Error args:\n" + "\n".join(map(str, args)) + "\nError kwargs:\n"
        )
        extra += "\n".join(f"{key}:{val}" for key, val in kwargs.items())
        print(extra)
        print(log_active_exception())
        await super().on_error(event, *args, **kwargs)


def run() -> None:
    """Run bot."""
    if TOKEN is None:
        print(
            """\nNo token set!
Either add ".env" file in bots folder with DISCORD_TOKEN=<token here> line,
or set DISCORD_TOKEN environment variable.""",
        )
        return
    print("\nStarting bot...")

    intents = discord.Intents(
        guild_messages=True,
        messages=True,
        guilds=True,
        guild_typing=True,
        message_content=True,
        voice_states=True,
    )

    loop = asyncio.new_event_loop()
    bot = MusicBot(intents=intents, loop=loop)

    try:
        loop.run_until_complete(bot.start(TOKEN))
    except KeyboardInterrupt:
        print("\nClosing bot...")
        loop.run_until_complete(bot.close())
    finally:
        # cancel all lingering tasks
        loop.close()
        print("\nBot has been deactivated.")


if __name__ == "__main__":
    print(f"{__title__} v{__version__}\nProgrammed by {__author__}.")
    run()
