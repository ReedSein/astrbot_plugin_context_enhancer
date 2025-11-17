"""
智能群聊上下文增强插件 v2.3 (最终兼容与安全版)
"""
import traceback
import json
import re
import datetime
import heapq
import itertools
from collections import deque, defaultdict
import os
from typing import Dict, Optional, Deque, List
from asyncio import Lock, TimerHandle
import time
import uuid
from dataclasses import dataclass
import asyncio
import aiofiles
import aiofiles.os as aio_os
from aiofiles.os import remove as aio_remove, rename as aio_rename

from astrbot.api.event import filter as event_filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api import logger, AstrBotConfig
from astrbot.api.provider import ProviderRequest
from astrbot.api.message_components import Plain, At, Image, Face, Reply
from astrbot.api.platform import MessageType

# --- 内部事件通信的特殊前缀 ---
INTERNAL_REPLY_PREFIX = "___INTERNAL_ACTIVE_REPLY___:::"

# 导入工具模块
try:
    from .utils.image_caption import ImageCaptionUtils
except ImportError:
    ImageCaptionUtils = None

class ContextMessageType:
    LLM_TRIGGERED = "llm_triggered"
    NORMAL_CHAT = "normal_chat"
    IMAGE_MESSAGE = "image_message"
    BOT_REPLY = "bot_reply"

class ContextConstants:
    MESSAGE_MATCH_TIME_WINDOW = 3
    PROMPT_HEADER = "你正在浏览聊天软件，查看群聊消息。"
    RECENT_CHATS_HEADER = "\n最近的聊天记录:"
    BOT_REPLIES_HEADER = "\n你最近的回复:"
    PROMPT_FOOTER = "请基于以上信息，并严格按照你的角色设定，做出自然且符合当前对话氛围的回复。"

@dataclass
class PluginConfig:
    enabled_groups: list
    recent_chats_count: int
    bot_replies_count: int
    collect_bot_replies: bool
    max_images_in_context: int
    enable_image_caption: bool
    image_caption_provider_id: str
    image_caption_prompt: str
    image_caption_timeout: int
    cleanup_interval_seconds: int
    inactive_cleanup_days: int
    command_prefixes: list
    duplicate_check_window_messages: int
    duplicate_check_time_seconds: int
    passive_reply_instruction: str
    active_speech_instruction: str
    enable_active_reply: bool
    active_reply_delay: int
    active_reply_min_messages: int
    active_reply_context_ttl: int
    active_reply_persona: str
    active_reply_prompt: str
    active_reply_probability: float

@dataclass
class GroupMessageBuffers:
    recent_chats: deque
    bot_replies: deque
    image_messages: deque

class GroupMessage:
    def __init__(self,
                 message_type: str,
                 sender_id: str,
                 sender_name: str,
                 group_id: str,
                 text_content: str = "",
                 images: Optional[list[str]] = None,
                 message_id: Optional[str] = None,
                 nonce: Optional[str] = None,
                 raw_components: Optional[list] = None):
        self.id = message_id
        self.nonce = nonce
        self.message_type = message_type
        self.timestamp = datetime.datetime.now()
        self.sender_id = sender_id
        self.sender_name = sender_name
        self.group_id = group_id
        self.text_content = text_content
        self.images = images or []
        self.has_image = len(self.images) > 0
        self.image_captions: list[str] = []
        self.raw_components = raw_components or []

    def to_dict(self) -> dict:
        serializable_components = []
        for comp in self.raw_components:
            if hasattr(comp, 'to_dict'):
                serializable_components.append(comp.to_dict())
            else:
                try:
                    serializable_components.append({"type": comp.__class__.__name__, "content": str(comp)})
                except Exception:
                    serializable_components.append({"type": "unknown", "content": str(comp)})
        return {
            "id": self.id, "nonce": self.nonce, "message_type": self.message_type,
            "timestamp": self.timestamp.isoformat(), "sender_name": self.sender_name,
            "sender_id": self.sender_id, "group_id": self.group_id, "text_content": self.text_content,
            "has_image": self.has_image, "image_captions": self.image_captions, "images": self.images,
            "raw_components": serializable_components
        }

    @classmethod
    def from_dict(cls, data: dict):
        instance = cls(
           message_type=data.get("message_type", ContextMessageType.NORMAL_CHAT),
           sender_id=data.get("sender_id", "unknown"),
           sender_name=data.get("sender_name", "用户"),
           group_id=data.get("group_id", ""),
           text_content=data.get("text_content", ""),
           images=data.get("images", []),
           message_id=data.get("id"),
           nonce=data.get("nonce"),
           raw_components=data.get("raw_components", [])
        )
        timestamp_str = data.get("timestamp")
        instance.timestamp = datetime.datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.datetime.now()
        instance.image_captions = data.get("image_captions", [])
        instance.has_image = len(instance.images) > 0
        return instance

@register("context_enhancer_v2", "木有知", "智能群聊上下文增强插件 v2.3", "2.3.0", repo="https://github.com/muyouzhi6/astrbot_plugin_context_enhancer")
class ContextEnhancerV2(Star):
    CACHE_LOAD_BUFFER_MULTIPLIER = 2

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context, config)
        self.raw_config = config
        self.config = self._load_plugin_config()
        self._global_lock = asyncio.Lock()
        logger.info("[ContextEnhancerV2] 上下文增强器v2.3已初始化")
        self._initialize_utils()
        self.group_messages: Dict[str, "GroupMessageBuffers"] = {}
        self.group_locks: defaultdict[str, Lock] = defaultdict(Lock)
        self.group_last_activity: Dict[str, datetime.datetime] = {}
        self.last_cleanup_time = time.time()
        self.data_dir = os.path.join(StarTools.get_data_dir(), "astrbot_plugin_context_enhancer")
        os.makedirs(self.data_dir, exist_ok=True)
        self.cache_path = os.path.join(self.data_dir, "context_cache.json")
        self.active_reply_timers: Dict[str, TimerHandle] = {}
        logger.info(f"上下文增强器配置加载完成: {self.config}")

    async def _async_init(self):
        await self._load_cache_from_file()
        logger.info(f"成功从 {self.cache_path} 异步加载上下文缓存")

    async def terminate(self):
        temp_path = self.cache_path + ".tmp"
        try:
            serializable_data = {}
            for group_id, buffers in self.group_messages.items():
                all_messages = list(heapq.merge(
                    buffers.recent_chats, buffers.bot_replies, buffers.image_messages, key=lambda msg: msg.timestamp
                ))
                max_messages_to_save = self.config.recent_chats_count + self.config.bot_replies_count
                if len(all_messages) > max_messages_to_save:
                    all_messages = all_messages[-max_messages_to_save:]
                serializable_data[group_id] = [msg.to_dict() for msg in all_messages]
            async with aiofiles.open(temp_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(serializable_data, ensure_ascii=False, indent=4))
            await aio_rename(temp_path, self.cache_path)
            logger.info(f"上下文缓存已成功原子化保存到 {self.cache_path}")
        except Exception as e:
            logger.error(f"[ContextEnhancerV2] 异步保存上下文缓存失败: {e}")
        finally:
            if await aio_os.path.exists(temp_path):
                try:
                    await aio_remove(temp_path)
                except Exception as e:
                    logger.error(f"[ContextEnhancerV2] 清理临时缓存文件 {temp_path} 失败: {e}")
        # 取消所有主动回复定时器
        for timer in self.active_reply_timers.values():
            timer.cancel()
        self.active_reply_timers.clear()
        logger.info("[ContextEnhancerV2] 所有主动回复定时器已取消。")

        if self.image_caption_utils and hasattr(self.image_caption_utils, 'close'):
            await self.image_caption_utils.close()

    def _load_plugin_config(self) -> PluginConfig:
        return PluginConfig(
            enabled_groups=[str(g) for g in self.raw_config.get("enabled_groups", [])],
            recent_chats_count=self.raw_config.get("recent_chats_count", 15),
            bot_replies_count=self.raw_config.get("bot_replies_count", 5),
            max_images_in_context=self.raw_config.get("max_context_images", 4),
            collect_bot_replies=self.raw_config.get("collect_bot_replies", True),
            enable_image_caption=self.raw_config.get("enable_image_caption", True),
            image_caption_provider_id=self.raw_config.get("image_caption_provider_id", ""),
            image_caption_prompt=self.raw_config.get("image_caption_prompt", "请简洁地描述这张图片的主要内容"),
            image_caption_timeout=self.raw_config.get("image_caption_timeout", 30),
            cleanup_interval_seconds=self.raw_config.get("cleanup_interval_seconds", 600),
            inactive_cleanup_days=self.raw_config.get("inactive_cleanup_days", 7),
            command_prefixes=self.raw_config.get("command_prefixes", ["/", "!", "！", "#", ".", "。"]),
            duplicate_check_window_messages=self.raw_config.get("duplicate_check_window_messages", 5),
            duplicate_check_time_seconds=self.raw_config.get("duplicate_check_time_seconds", 30),
            passive_reply_instruction=self.raw_config.get("passive_reply_instruction", '现在，群成员 {sender_name} (ID: {sender_id}) 正在对你说话，TA说："{original_prompt}"'),
            active_speech_instruction=self.raw_config.get("active_speech_instruction", '以上是最近的聊天记录。你决定主动参与讨论，并想就以下内容发表你的看法："{original_prompt}"'),
            enable_active_reply=self.raw_config.get("enable_active_reply", False),
            active_reply_delay=self.raw_config.get("active_reply_delay", 120),
            active_reply_min_messages=self.raw_config.get("active_reply_min_messages", 10),
            active_reply_context_ttl=self.raw_config.get("active_reply_context_ttl", 300),
            active_reply_persona=self.raw_config.get("active_reply_persona", ""),
            active_reply_prompt=self.raw_config.get("active_reply_prompt", "请你分析以上聊天记录，判断当前最热门或最有趣的话题是什么，并主动发表你的看法，自然地融入对话。"),
            active_reply_probability=self.raw_config.get("active_reply_probability", 0.3)
        )

    def _initialize_utils(self):
        try:
            if ImageCaptionUtils is not None:
                self.image_caption_utils = ImageCaptionUtils(self.context, self.raw_config)
            else:
                self.image_caption_utils = None
                logger.warning("[ContextEnhancerV2] ImageCaptionUtils 不可用")
        except Exception as e:
            logger.error(f"[ContextEnhancerV2] 工具类初始化失败: {e}")
            self.image_caption_utils = None

    def _get_or_create_lock(self, group_id: str) -> Lock:
        return self.group_locks[group_id]

    async def _load_cache_from_file(self):
        if not await aio_os.path.exists(self.cache_path): return
        try:
            async with aiofiles.open(self.cache_path, "r", encoding="utf-8") as f:
                content = await f.read()
                if content:
                    data = json.loads(content)
                    self.group_messages = self._load_group_messages_from_dict(data)
        except Exception as e:
            logger.error(f"[ContextEnhancerV2] 异步加载上下文缓存失败: {e}")

    def _load_group_messages_from_dict(self, data: Dict[str, list]) -> Dict[str, "GroupMessageBuffers"]:
        group_buffers_map = {}
        for group_id, msg_list in data.items():
            buffers = self._create_new_group_buffers()
            for msg_data in msg_list:
                try:
                    msg = GroupMessage.from_dict(msg_data)
                    if msg.message_type == ContextMessageType.BOT_REPLY: buffers.bot_replies.append(msg)
                    elif msg.has_image: buffers.image_messages.append(msg)
                    else: buffers.recent_chats.append(msg)
                except Exception as e:
                    logger.warning(f"[ContextEnhancerV2] 从字典转换消息失败: {e}")
            group_buffers_map[group_id] = buffers
        return group_buffers_map

    def _create_new_group_buffers(self) -> "GroupMessageBuffers":
        return GroupMessageBuffers(
            recent_chats=deque(maxlen=self.config.recent_chats_count * self.CACHE_LOAD_BUFFER_MULTIPLIER),
            bot_replies=deque(maxlen=self.config.bot_replies_count * self.CACHE_LOAD_BUFFER_MULTIPLIER),
            image_messages=deque(maxlen=self.config.max_images_in_context * self.CACHE_LOAD_BUFFER_MULTIPLIER)
        )

    async def _get_or_create_group_buffers(self, group_id: str) -> "GroupMessageBuffers":
        current_dt = datetime.datetime.now()
        self.group_last_activity[group_id] = current_dt
        now = time.time()
        if now - self.last_cleanup_time > self.config.cleanup_interval_seconds:
            await self._cleanup_inactive_groups(current_dt)
            self.last_cleanup_time = now
        if group_id not in self.group_messages:
            async with self._global_lock:
                if group_id not in self.group_messages:
                    self.group_messages[group_id] = self._create_new_group_buffers()
        return self.group_messages[group_id]

    async def _cleanup_inactive_groups(self, current_time: datetime.datetime):
        inactive_threshold = datetime.timedelta(days=self.config.inactive_cleanup_days)
        inactive_groups = [gid for gid, lact in list(self.group_last_activity.items()) if current_time - lact > inactive_threshold]
        if inactive_groups:
            async with self._global_lock:
                for group_id in inactive_groups:
                    self.group_messages.pop(group_id, None)
                    self.group_last_activity.pop(group_id, None)
                    self.group_locks.pop(group_id, None)
            logger.info(f"清理了 {len(inactive_groups)} 个不活跃群组的上下文。")

    def is_chat_enabled(self, event: AstrMessageEvent) -> bool:
        if event.get_message_type() == MessageType.FRIEND_MESSAGE: return True
        group_id = event.get_group_id()
        return not self.config.enabled_groups or group_id in self.config.enabled_groups

    @event_filter.platform_adapter_type(event_filter.PlatformAdapterType.ALL, priority=0)
    async def on_message(self, event: AstrMessageEvent):
        # --- 架构重构：合并内部事件处理与常规消息处理 ---
        # 1. 首先检查是否是我们的内部事件
        if self._is_bot_message(event) and event.get_plain_text().startswith(INTERNAL_REPLY_PREFIX):
            logger.debug("[ContextEnhancerV2] 捕获到主动回复内部事件。")
            event.stop_propagation()
            event.stop_event()
            payload = event.get_plain_text().replace(INTERNAL_REPLY_PREFIX, "", 1)
            result = MessageEventResult()
            result.message(payload)
            event.set_result(result)
            logger.debug("[ContextEnhancerV2] 内部事件已转换为标准结果，送入流水线处理。")
            return # 处理完毕，直接返回

        # 2. 如果不是内部事件，再执行常规的消息收集逻辑
        if not self.is_chat_enabled(event): return
        
        message_text = (event.message_str or "").strip()
        if message_text.lower() in ["reset", "new"]:
            await self.handle_clear_context_command(event)
            return
            
        if event.get_message_type() == MessageType.GROUP_MESSAGE:
            await self._handle_group_message(event)

    def _extract_user_info_from_event(self, event: AstrMessageEvent) -> tuple[str, str]:
        sender_name = event.get_sender_name()
        sender_id = event.get_sender_id()
        if not sender_name or not sender_id:
            message_obj = getattr(event, 'message_obj', None)
            if message_obj and hasattr(message_obj, 'sender') and message_obj.sender:
                sender = message_obj.sender
                if not sender_name and hasattr(sender, 'nickname'): sender_name = sender.nickname
                if not sender_id and hasattr(sender, 'user_id'): sender_id = str(sender.user_id)
        if not sender_name or not sender_id:
            raw_event = getattr(event, 'raw_event', None)
            if raw_event and isinstance(raw_event.get("sender"), dict):
                raw_sender = raw_event["sender"]
                if not sender_name: sender_name = raw_sender.get("card") or raw_sender.get("nickname")
                if not sender_id: sender_id = str(raw_sender.get("user_id") or raw_sender.get("id"))
        return sender_name or "用户", sender_id or "unknown"

    async def _get_image_captions(self, images: list[str]) -> list[str]:
        if not self.config.enable_image_caption or not self.image_caption_utils:
            return ["图片"] * len(images)
        tasks = [self.image_caption_utils.generate_image_caption(
            url, timeout=self.config.image_caption_timeout,
            provider_id=self.config.image_caption_provider_id or None,
            custom_prompt=self.config.image_caption_prompt
        ) for url in images if url]
        if not tasks: return []
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [res if not isinstance(res, Exception) else "图片内容未知" for res in results]

    async def _create_group_message_from_event(self, event: AstrMessageEvent, message_type: str) -> GroupMessage:
        text_parts, images = [], []
        raw_components = getattr(event.message_obj, 'message', [])
        for comp in raw_components:
            if isinstance(comp, Plain): text_parts.append(comp.text)
            elif isinstance(comp, At): text_parts.append(f"@{comp.qq}")
            elif isinstance(comp, Face): text_parts.append(f"[表情]")
            elif isinstance(comp, Reply): text_parts.append(f"[引用]")
            elif isinstance(comp, Image):
                url = getattr(comp, "url", None) or getattr(comp, "file", None)
                if url: images.append(url)
        if images:
            captions = await self._get_image_captions(images)
            text_parts.append(f"[Image: {'; '.join(captions)}]")
        sender_name, sender_id = self._extract_user_info_from_event(event)
        return GroupMessage(
            message_type=message_type, sender_id=sender_id, sender_name=sender_name,
            group_id=event.get_group_id(), text_content="".join(text_parts).strip(),
            images=images, message_id=getattr(event, 'id', None) or getattr(event.message_obj, 'id', None),
            nonce=getattr(event, '_context_enhancer_nonce', None), raw_components=raw_components
        )
        
    async def _handle_group_message(self, event: AstrMessageEvent):
        group_msg = await self._create_group_message_from_event(event, "")
        if not group_msg.text_content and not group_msg.has_image: return
        message_type = self._classify_message(event)
        group_msg.message_type = message_type
        buffers = await self._get_or_create_group_buffers(group_msg.group_id)
        lock = self._get_or_create_lock(group_msg.group_id)
        async with lock:
            target_deque = buffers.bot_replies if message_type == ContextMessageType.BOT_REPLY else buffers.recent_chats
            if not self._is_duplicate_message(target_deque, group_msg):
                target_deque.append(group_msg)
        self._schedule_active_reply(event)

    def _is_duplicate_message(self, target_deque: deque, new_msg: GroupMessage) -> bool:
        if new_msg.has_image: return False
        start_index = max(0, len(target_deque) - self.config.duplicate_check_window_messages)
        recent_messages = list(itertools.islice(target_deque, start_index, len(target_deque)))
        for msg in recent_messages:
            if (msg.sender_id == new_msg.sender_id and msg.text_content == new_msg.text_content and
                    abs((new_msg.timestamp - msg.timestamp).total_seconds()) < self.config.duplicate_check_time_seconds):
                return True
        return False

    def _is_bot_message(self, event: AstrMessageEvent) -> bool:
        try:
            return bool(event.get_self_id() and event.get_sender_id() and str(event.get_sender_id()) == str(event.get_self_id()))
        except (AttributeError, KeyError): return False

    def _classify_message(self, event: AstrMessageEvent) -> str:
        if self._is_bot_message(event) and self.config.bot_replies_count > 0:
            return ContextMessageType.BOT_REPLY
        if self._is_directly_triggered(event):
            setattr(event, '_context_enhancer_nonce', uuid.uuid4().hex)
            return ContextMessageType.LLM_TRIGGERED
        return ContextMessageType.NORMAL_CHAT

    def _is_at_triggered(self, event: AstrMessageEvent) -> bool:
        bot_id = event.get_self_id()
        if not bot_id: return False
        if event.message_obj and event.message_obj.message:
            for comp in event.message_obj.message:
                if isinstance(comp, At) and (str(comp.qq) == str(bot_id) or comp.qq == "all"): return True
        return False

    def _is_keyword_triggered(self, event: AstrMessageEvent) -> bool:
        message_text = (event.message_str or "").lower().strip()
        if not message_text: return False
        return any(message_text.startswith(prefix) for prefix in self.config.command_prefixes)

    def _is_directly_triggered(self, event: AstrMessageEvent) -> bool:
        return self._is_at_triggered(event) or self._is_keyword_triggered(event)

    @event_filter.on_llm_request(priority=100)
    async def on_llm_request(self, event: AstrMessageEvent, request: ProviderRequest):
        if not self._should_enhance_context(event, request): return
        group_id = event.get_group_id()
        buffers = await self._get_or_create_group_buffers(group_id)
        if not any([buffers.recent_chats, buffers.bot_replies, buffers.image_messages]): return
        lock = self._get_or_create_lock(group_id)
        async with lock:
            all_messages = list(heapq.merge(buffers.recent_chats, buffers.bot_replies, buffers.image_messages, key=lambda x: x.timestamp))
            triggering_message, scene = self._find_triggering_message_from_event(all_messages, event)
            context_enhancement, image_urls_for_context = self._build_context_enhancement(
                all_messages, request.prompt, triggering_message, scene, event
            )
        self._inject_context_into_request(request, context_enhancement, image_urls_for_context)

    def _should_enhance_context(self, event: AstrMessageEvent, request: ProviderRequest) -> bool:
        return (
            not hasattr(request, '_context_enhanced') and self.is_chat_enabled(event) and
            event.get_message_type() == MessageType.GROUP_MESSAGE
        )

    def _extract_messages_for_context(self, sorted_messages: list[GroupMessage]) -> dict:
        max_chats, max_bot_replies = self.config.recent_chats_count, self.config.bot_replies_count
        bot_replies = [f"你回复了: {msg.text_content}" for msg in itertools.islice(
            (m for m in reversed(sorted_messages) if m.message_type == ContextMessageType.BOT_REPLY), max_bot_replies)]
        recent_chats = [f"{msg.sender_name}: {msg.text_content}" for msg in itertools.islice(
            (m for m in reversed(sorted_messages) if m.message_type != ContextMessageType.BOT_REPLY and m.text_content), max_chats)]
        return {"recent_chats": list(reversed(recent_chats)), "bot_replies": list(reversed(bot_replies))}

    def _build_context_enhancement(self, sorted_messages: list[GroupMessage], original_prompt: str,
                                  triggering_message: Optional[GroupMessage], scene: str, event: AstrMessageEvent) -> tuple[str, list[str]]:
        extracted_data = self._extract_messages_for_context(sorted_messages)
        image_urls = [url for msg in sorted_messages if msg.images for url in msg.images]
        if len(image_urls) > self.config.max_images_in_context:
            image_urls = image_urls[-self.config.max_images_in_context:]
        history_parts = [ContextConstants.PROMPT_HEADER]
        history_parts.extend(self._format_recent_chats_section(extracted_data["recent_chats"]))
        history_parts.extend(self._format_bot_replies_section(extracted_data["bot_replies"]))
        context_str = "\n".join(part for part in history_parts if part)
        instruction_prompt = self._format_situation_instruction(original_prompt, triggering_message, scene, event)
        return f"{context_str}\n\n{instruction_prompt}", image_urls

    def _inject_context_into_request(self, request: ProviderRequest, context_enhancement: str, image_urls: list[str]):
    """注入上下文增强和图片到请求中
    
    Args:
        request: 提供商请求对象
        context_enhancement: 增强的上下文文本
        image_urls: 图片URL列表
    """
    from astrbot.core.agent.message import TextPart, ImageURLPart
    
    # 首先处理文本上下文
    if context_enhancement:
        # 如果有图片要添加，我们稍后会一起处理
        # 如果没有图片，直接设置 prompt
        if not image_urls:
            request.prompt = context_enhancement
            setattr(request, '_context_enhanced', True)
    
    # 处理图片
    if image_urls and len(image_urls) > 0:
        try:
            # 限制图片数量
            limited_image_urls = image_urls[-self.config.max_images_in_context:] if len(image_urls) > self.config.max_images_in_context else image_urls
            
            # 构造 content 列表
            content_parts = []
            
            # 添加文本部分（使用增强后的上下文）
            text_content = context_enhancement if context_enhancement else request.prompt
            if text_content:
                content_parts.append(TextPart(text=text_content))
            
            # 添加图片部分 - 使用 Pydantic 模型
            for img_url in limited_image_urls:
                if img_url:
                    try:
                        # 构造符合 ImageURLPart 的对象
                        content_parts.append(
                            ImageURLPart(
                                image_url=ImageURLPart.ImageURL(url=img_url)
                            )
                        )
                    except Exception as e:
                        logger.warning(f"[ContextEnhancerV2] 添加图片 {img_url[:50]}... 失败: {e}")
            
            # 如果成功添加了内容
            if content_parts:
                # 创建一个新的用户消息，包含文本和图片
                user_message = {
                    "role": "user",
                    "content": content_parts
                }
                
                # 将这个消息添加到 contexts 的末尾
                request.contexts.append(user_message)
                
                # 清空 prompt，让系统使用 contexts 中的消息
                request.prompt = None
                
                setattr(request, '_context_enhanced', True)
                logger.debug(f"[ContextEnhancerV2] 已添加 {len(limited_image_urls)} 张图片和上下文文本到请求中")
            
        except Exception as e:
            logger.error(f"[ContextEnhancerV2] 处理图片时发生错误: {e}")
            # 出错时回退到只使用文本
            if context_enhancement:
                request.prompt = context_enhancement
                setattr(request, '_context_enhanced', True)
    elif context_enhancement:
        # 没有图片时，直接设置 prompt
        request.prompt = context_enhancement
        setattr(request, '_context_enhanced', True)

    def _find_triggering_message_from_event(self, sorted_messages: list[GroupMessage], llm_request_event: AstrMessageEvent) -> tuple[Optional[GroupMessage], str]:
        nonce = getattr(llm_request_event, '_context_enhancer_nonce', None)
        if not nonce: return None, "主动发言"
        trigger_message = next((msg for msg in reversed(sorted_messages) if msg.nonce == nonce), None)
        return trigger_message, "被动回复"

    def _format_recent_chats_section(self, recent_chats: list) -> list:
        if not recent_chats: return []
        return [ContextConstants.RECENT_CHATS_HEADER] + recent_chats

    def _format_bot_replies_section(self, bot_replies: list) -> list:
        if not bot_replies: return []
        return [ContextConstants.BOT_REPLIES_HEADER] + bot_replies

    def _format_situation_instruction(self, original_prompt: str, triggering_message: Optional[GroupMessage],
                                     scenario: str, event: AstrMessageEvent) -> str:
        if scenario == "被动回复":
            instruction = self.config.passive_reply_instruction
            if triggering_message:
                sender_name, sender_id = triggering_message.sender_name, triggering_message.sender_id
            else:
                sender_name, sender_id = self._extract_user_info_from_event(event)
            return instruction.format(sender_name=sender_name, sender_id=sender_id, original_prompt=original_prompt)
        else:
            return self.config.active_speech_instruction.format(original_prompt=original_prompt)

    @event_filter.on_llm_response(priority=100)
    async def on_llm_response(self, event: AstrMessageEvent, resp):
        if event.get_message_type() == MessageType.GROUP_MESSAGE:
            group_id = event.get_group_id()
            response_text = ""
            if hasattr(resp, "completion_text"): response_text = resp.completion_text
            elif hasattr(resp, "text"): response_text = resp.text
            else: response_text = str(resp)
            bot_reply = GroupMessage(
                message_type=ContextMessageType.BOT_REPLY, sender_id=event.get_self_id(),
                sender_name=self.raw_config.get("name", "助手"), group_id=group_id,
                text_content=response_text[:1000]
            )
            buffers = await self._get_or_create_group_buffers(group_id)
            lock = self._get_or_create_lock(group_id)
            async with lock: buffers.bot_replies.append(bot_reply)

    async def clear_context_cache(self, group_id: Optional[str] = None):
        if group_id:
            if group_id in self.group_messages:
                lock = self._get_or_create_lock(group_id)
                async with lock:
                    self.group_messages.pop(group_id, None)
                    self.group_locks.pop(group_id, None)
                    self.group_last_activity.pop(group_id, None)
        else:
            async with self._global_lock: self.group_messages.clear()
            self.group_last_activity.clear()
            if await aio_os.path.exists(self.cache_path): await aio_remove(self.cache_path)

    @event_filter.command("reset", "new", description="清空当前群聊的上下文缓存")
    async def handle_clear_context_command(self, event: AstrMessageEvent):
        group_id = event.get_group_id()
        if group_id: await self.clear_context_cache(group_id=group_id)

    def _schedule_active_reply(self, event: AstrMessageEvent):
        if not self.config.enable_active_reply: return
        group_id = event.get_group_id()
        if not group_id: return
        if group_id in self.active_reply_timers:
            self.active_reply_timers[group_id].cancel()
        loop = asyncio.get_event_loop()
        self.active_reply_timers[group_id] = loop.call_later(
            self.config.active_reply_delay,
            lambda: loop.create_task(self._check_and_trigger_active_reply(event, group_id)),
        )

    async def _check_and_trigger_active_reply(self, event: AstrMessageEvent, group_id: str):
        if random.random() > self.config.active_reply_probability:
            return
        await self._perform_active_reply(event, group_id)

    async def _perform_active_reply(self, event: AstrMessageEvent, group_id: str):
        async with self._get_or_create_lock(group_id):
            if group_id not in self.active_reply_timers:
                return
            del self.active_reply_timers[group_id]
        logger.info(f"[ContextEnhancerV2] 在群 {group_id} 触发主动回复检查。")
        persona_id = self.config.active_reply_persona
        persona = None
        if persona_id:
            try: persona = await self.context.persona_manager.get_persona(persona_id)
            except Exception as e: logger.warning(f"获取主动回复人格 {persona_id} 失败: {e}")
        buffers = await self._get_or_create_group_buffers(group_id)
        all_messages = list(heapq.merge(
            buffers.recent_chats, buffers.bot_replies, key=lambda x: x.timestamp))
        time_window = datetime.timedelta(seconds=self.config.active_reply_context_ttl)
        now = datetime.datetime.now()
        context_messages = [msg for msg in all_messages if now - msg.timestamp < time_window]
        if len(context_messages) < self.config.active_reply_min_messages:
            logger.debug(f"群 {group_id} 近期消息不足，取消主动回复。")
            return
        formatted_context = "\n".join([f"{msg.sender_name}: {msg.text_content}" for msg in context_messages])
        prompt = self.config.active_reply_prompt.replace("{context}", formatted_context)
        try:
            provider = self.context.get_using_provider()
            if not provider: return
            kwargs = {"prompt": prompt}
            if persona and persona.system_prompt:
                kwargs["system_prompt"] = persona.system_prompt
            
            resp = await provider.text_chat(**kwargs)
            
            if resp and resp.completion_text:
                logger.debug(f"主动回复原始文本: {resp.completion_text[:100]}...")
                internal_message = f"{INTERNAL_REPLY_PREFIX}{resp.completion_text}"
                await self.context.send_message(
                    message=internal_message,
                    target_id=group_id,
                    target_type="group",
                    platform=event.get_platform_name(),
                )
        except Exception as e:
            logger.error(f"执行主动回复时发生严重错误: {e}", exc_info=True)
