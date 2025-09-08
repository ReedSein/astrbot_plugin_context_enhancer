"""
智能群聊上下文增强插件
通过多维度信息收集和分层架构，为 LLM 提供丰富的群聊语境，支持角色扮演，完全兼容人设系统。
"""
import traceback
import json
import datetime
from collections import deque
import os
from typing import Dict, Optional
import time
import uuid
from dataclasses import dataclass
import asyncio
import aiofiles

from astrbot.api.event import filter as event_filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api import logger, AstrBotConfig
from astrbot.api.provider import ProviderRequest
from astrbot.api.message_components import Plain, At, Image
from astrbot.api.platform import MessageType

# 导入工具模块
try:
    from .utils.image_caption import ImageCaptionUtils
    from .utils.message_utils import MessageUtils
except ImportError:
    ImageCaptionUtils = None
    MessageUtils = None
    # _initialize_utils 方法中会记录详细日志


# 消息类型枚举 - 重命名以避免冲突
class ContextMessageType:
    """消息类型枚举"""
    LLM_TRIGGERED = "llm_triggered"
    NORMAL_CHAT = "normal_chat"
    IMAGE_MESSAGE = "image_message"
    BOT_REPLY = "bot_reply"


# 常量定义 - 避免硬编码
class ContextConstants:
    """插件中使用的常量"""
    MESSAGE_MATCH_TIME_WINDOW = 3
    PROMPT_HEADER = "你正在浏览聊天软件，查看群聊消息。"
    RECENT_CHATS_HEADER = "\n最近的聊天记录:"
    BOT_REPLIES_HEADER = "\n你最近的回复:"
    USER_TRIGGER_TEMPLATE = "\n现在 {sender_name}（ID: {sender_id}）发了一个消息: {original_prompt}"
    PROACTIVE_TRIGGER_TEMPLATE = "\n你需要根据以上聊天记录，主动就以下内容发表观点: {original_prompt}"
    PROMPT_FOOTER = "需要你在心里理清当前到底讨论的什么，搞清楚形势，谁在跟谁说话，你是在插话还是回复，然后根据你的设定和当前形势做出最自然的回复。"


@dataclass
class PluginConfig:
    """统一管理插件配置项"""
    enabled_groups: list
    recent_chats_count: int
    bot_replies_count: int
    max_images_in_context: int
    enable_image_caption: bool
    image_caption_provider_id: str
    image_caption_prompt: str
    cleanup_interval_seconds: int
    inactive_cleanup_days: int
    command_prefixes: list
    duplicate_check_window_messages: int
    duplicate_check_time_seconds: int


class GroupMessage:
    """群聊消息的独立数据类，与框架解耦"""
    def __init__(self,
                 message_type: str,
                 sender_id: str,
                 sender_name: str,
                 group_id: str,
                 text_content: str = "",
                 images: Optional[list] = None,
                 message_id: Optional[str] = None,
                 nonce: Optional[str] = None):
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

    def to_dict(self) -> dict:
        """将消息对象转换为可序列化为 JSON 的字典"""
        return {
            "id": self.id,
            "nonce": self.nonce,
            "message_type": self.message_type,
            "timestamp": self.timestamp.isoformat(),
            "sender_name": self.sender_name,
            "sender_id": self.sender_id,
            "group_id": self.group_id,
            "text_content": self.text_content,
            "has_image": self.has_image,
            "image_captions": self.image_captions,
            "image_urls": [getattr(img, "url", None) or getattr(img, "file", None) for img in self.images]
        }

    @classmethod
    def from_dict(cls, data: dict):
        """从字典创建 GroupMessage 对象"""
        instance = cls(
            message_type=data["message_type"],
            sender_id=data.get("sender_id", "unknown"),
            sender_name=data.get("sender_name", "用户"),
            group_id=data.get("group_id", ""),
            text_content=data.get("text_content", ""),
            images=[Image.fromURL(url=url) for url in data.get("image_urls", []) if url],
            message_id=data.get("id"),
            nonce=data.get("nonce")
        )
        instance.timestamp = datetime.datetime.fromisoformat(data["timestamp"])
        instance.image_captions = data.get("image_captions", [])
        return instance




@register(
    "astrbot_plugin_context_enhancer",
    "木有知",
    "智能群聊上下文增强插件v2.0，提供强大的'读空气'功能。通过多维度信息收集和分层架构，为 LLM 提供丰富的群聊语境，支持角色扮演，完全兼容人设系统。",
    "2.0.0",
)
class ContextEnhancerV2(Star):
    """
    AstrBot 上下文增强器 v2.0

    作者: 木有知 (https://github.com/muyouzhi6)

    功能特点:
    - 🎯 简单直接的上下文增强，参考SpectreCore的简洁方式
    - 📝 自动收集群聊历史和机器人回复记录
    - 🖼️ 支持图片描述和高级消息格式化（可选）
    - 🛡️ 安全兼容，不覆盖system_prompt，不干扰其他插件

    技术保证:
    - 不影响 system_prompt，完全兼容人设系统
    - 使用合理优先级，不干扰其他插件
    - 异步处理，不阻塞主流程
    - 完善的错误处理和功能降级
    """
    # 缓冲区大小乘数，用于为 deque 提供额外空间，避免在消息快速增长时频繁丢弃旧消息
    CACHE_LOAD_BUFFER_MULTIPLIER = 2

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context, config)
        self.raw_config = config
        self.config = self._load_plugin_config()
        logger.info("上下文增强器v2.0已初始化")

        # 初始化工具类
        self._initialize_utils()

        # 群聊消息缓存 - 每个群独立存储
        self.group_messages: Dict[str, deque[GroupMessage]] = {}
        self.group_last_activity: Dict[str, datetime.datetime] = {}
        self.last_cleanup_time = time.time()

        # 异步加载持久化的上下文
        self.data_dir = os.path.join(
            StarTools.get_data_dir(), "astrbot_plugin_context_enhancer"
        )
        os.makedirs(self.data_dir, exist_ok=True)
        self.cache_path = os.path.join(self.data_dir, "context_cache.json")
        
        # 显示当前配置
        logger.info(f"上下文增强器配置加载完成: {self.config}")

    async def _async_init(self):
        """异步初始化部分，例如加载缓存"""
        await self._load_cache_from_file()

    async def terminate(self):
        """插件终止时，异步持久化上下文并关闭会话"""
        # 异步持久化上下文
        try:
            serializable_messages = {}
            for group_id, messages in self.group_messages.items():
                serializable_messages[group_id] = [msg.to_dict() for msg in messages]

            async with aiofiles.open(self.cache_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(serializable_messages, ensure_ascii=False, indent=4))
            logger.info(f"上下文缓存已成功异步保存到 {self.cache_path}")
        except Exception as e:
            logger.error(f"异步保存上下文缓存失败: {e}")

        # 关闭 aiohttp session
        if self.image_caption_utils and hasattr(self.image_caption_utils, 'close'):
            await self.image_caption_utils.close()
            logger.info("ImageCaptionUtils 的 aiohttp session 已关闭。")

    def _load_plugin_config(self) -> PluginConfig:
        """从原始配置加载并填充插件配置类"""
        return PluginConfig(
            enabled_groups=self.raw_config.get("启用群组", []),
            recent_chats_count=self.raw_config.get("最近聊天记录数量", 15),
            bot_replies_count=self.raw_config.get("机器人回复数量", 5),
            max_images_in_context=self.raw_config.get("上下文图片最大数量", 4),
            enable_image_caption=self.raw_config.get("启用图片描述", True),
            image_caption_provider_id=self.raw_config.get("图片描述提供商ID", ""),
            image_caption_prompt=self.raw_config.get(
                "图片描述提示词", "请简洁地描述这张图片的主要内容，重点关注与聊天相关的信息"
            ),
            cleanup_interval_seconds=self.raw_config.get("cleanup_interval_seconds", 600),
            inactive_cleanup_days=self.raw_config.get("inactive_cleanup_days", 7),
            command_prefixes=self.raw_config.get("command_prefixes", ["/", "!", "！", "#", ".", "。"]),
            duplicate_check_window_messages=self.raw_config.get("duplicate_check_window_messages", 5),
            duplicate_check_time_seconds=self.raw_config.get("duplicate_check_time_seconds", 30),
        )

    def _initialize_utils(self):
        """初始化工具模块"""
        try:
            if ImageCaptionUtils is not None:
                self.image_caption_utils = ImageCaptionUtils(
                    self.context, self.raw_config
                )
                logger.debug("ImageCaptionUtils 初始化成功")
            else:
                self.image_caption_utils = None
                logger.warning("ImageCaptionUtils 不可用，将使用基础图片处理")

            if MessageUtils is not None and self.image_caption_utils is not None:
                self.message_utils = MessageUtils(self.raw_config, self.context, self.image_caption_utils)
                logger.debug("MessageUtils 初始化成功")
            else:
                self.message_utils = None
                logger.warning("MessageUtils 不可用（或其依赖项 ImageCaptionUtils 不可用），将使用基础消息格式化")
        except Exception as e:
            logger.error(f"工具类初始化失败: {e}")
            self.image_caption_utils = None
            self.message_utils = None

    async def _load_cache_from_file(self):
        """从文件异步加载缓存"""
        if not os.path.exists(self.cache_path):
            return
        try:
            async with aiofiles.open(self.cache_path, "r", encoding="utf-8") as f:
                content = await f.read()
                if content: # 确保文件内容不为空
                    data = json.loads(content)
                    self.group_messages = self._load_group_messages_from_dict(data)
                    logger.info(f"成功从 {self.cache_path} 异步加载上下文缓存。")
                else:
                    logger.info(f"缓存文件 {self.cache_path} 为空，跳过加载。")
        except Exception as e:
            logger.error(f"异步加载上下文缓存失败: {e}")

    def _load_group_messages_from_dict(
        self, data: Dict[str, list]
    ) -> Dict[str, deque]:
        """从字典加载群组消息"""
        group_messages = {}

        # 计算 maxlen
        base_max_len = self.config.recent_chats_count + self.config.bot_replies_count
        max_len = base_max_len * self.CACHE_LOAD_BUFFER_MULTIPLIER

        for group_id, msg_list in data.items():
            # 为每个群组创建一个有最大长度限制的 deque
            message_deque = deque(maxlen=max_len)
            for msg_data in msg_list:
                try:
                    # 从字典重建 GroupMessage 对象
                    message_deque.append(GroupMessage.from_dict(msg_data))
                except Exception as e:
                    logger.warning(f"从字典转换消息失败 (群 {group_id}): {e}")
            group_messages[group_id] = message_deque
        return group_messages

    def _get_group_buffer(self, group_id: str) -> deque:
        """获取群聊的消息缓冲区，并管理内存"""
        current_dt = datetime.datetime.now()

        # 更新活动时间
        self.group_last_activity[group_id] = current_dt

        # 基于时间的缓存清理
        now = time.time()
        if now - self.last_cleanup_time > self.config.cleanup_interval_seconds:
            self._cleanup_inactive_groups(current_dt)
            self.last_cleanup_time = now

        if group_id not in self.group_messages:
            # 优化 maxlen 计算逻辑，使其与实际上下文使用的配置项关联
            max_len = (self.config.recent_chats_count + self.config.bot_replies_count) * self.CACHE_LOAD_BUFFER_MULTIPLIER
            self.group_messages[group_id] = deque(maxlen=max_len)
        return self.group_messages[group_id]

    def _cleanup_inactive_groups(self, current_time: datetime.datetime):
        """清理超过配置天数未活跃的群组缓存"""
        inactive_threshold = datetime.timedelta(
            days=self.config.inactive_cleanup_days
        )
        inactive_groups = []

        for group_id, last_activity in self.group_last_activity.items():
            if current_time - last_activity > inactive_threshold:
                inactive_groups.append(group_id)

        for group_id in inactive_groups:
            if group_id in self.group_messages:
                del self.group_messages[group_id]
            del self.group_last_activity[group_id]

        if inactive_groups:
            logger.debug(f"清理了 {len(inactive_groups)} 个不活跃群组的缓存")

    def is_chat_enabled(self, event: AstrMessageEvent) -> bool:
        """检查当前聊天是否启用增强功能"""
        if event.get_message_type() == MessageType.FRIEND_MESSAGE:
            return True  # 简化版本默认启用私聊
        else:
            group_id = event.get_group_id()
            logger.debug(f"群聊启用检查: 群ID={group_id}, 启用列表={self.config.enabled_groups}")

            if not self.config.enabled_groups:  # 空列表表示对所有群生效
                logger.debug("空的启用列表，对所有群生效")
                return True

            result = group_id in self.config.enabled_groups
            logger.debug(f"群聊启用结果: {result}")
            return result

    @event_filter.platform_adapter_type(event_filter.PlatformAdapterType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """监听所有消息，进行分类和存储"""
        try:
            if not self.is_chat_enabled(event):
                return

            # 检查是否是 reset 命令
            message_text = (event.message_str or "").strip()
            if message_text.lower() in ["reset", "new"]:
                await self.handle_clear_context_command(event)
                return

            if event.get_message_type() == MessageType.GROUP_MESSAGE:
                await self._handle_group_message(event)

        except Exception as e:
            logger.error(f"处理消息时发生错误: {e}")
            logger.error(traceback.format_exc())

    def _create_group_message_from_event(self, event: AstrMessageEvent, message_type: str) -> GroupMessage:
        """从事件创建 GroupMessage 实例"""
        text_content_parts = []
        images = []
        if event.message_obj and event.message_obj.message:
            for comp in event.message_obj.message:
                if isinstance(comp, Plain):
                    text_content_parts.append(comp.text)
                elif isinstance(comp, At):
                    text_content_parts.append(f"@{comp.qq}")
                elif isinstance(comp, Image):
                    images.append(comp)
        
        return GroupMessage(
            message_type=message_type,
            sender_id=event.get_sender_id() or "unknown",
            sender_name=event.get_sender_name() or "用户",
            group_id=event.get_group_id() or event.unified_msg_origin,
            text_content="".join(text_content_parts).strip(),
            images=images,
            message_id=getattr(event, 'id', None) or getattr(event.message_obj, 'id', None),
            nonce=getattr(event, '_context_enhancer_nonce', None)
        )

    async def _handle_group_message(self, event: AstrMessageEvent):
        """处理群聊消息"""
        group_msg = self._create_group_message_from_event(event, "")  # 临时创建以检查内容
        if not group_msg.text_content and not group_msg.images:
            logger.debug("消息为空（无文本无图片），跳过处理。")
            return

        try:
            if self._is_bot_message(event):
                logger.debug("收集到机器人自己的消息，用于保持上下文完整性。")

            message_type = self._classify_message(event)
            group_msg.message_type = message_type # 更新消息类型

            # 生成图片描述
            if group_msg.has_image and self.config.enable_image_caption:
                await self._generate_image_captions(group_msg)

            # 添加到缓冲区前进行去重检查
            buffer = self._get_group_buffer(group_msg.group_id)

            # 🚨 防重复机制：检查是否已存在相同消息
            if not self._is_duplicate_message(buffer, group_msg):
                buffer.append(group_msg)
                logger.debug(
                    f"收集群聊消息 [{message_type}]: {group_msg.sender_name} - {group_msg.text_content[:50]}..."
                )
            else:
                logger.debug(
                    f"跳过重复消息: {group_msg.sender_name} - {group_msg.text_content[:30]}..."
                )

        except Exception as e:
            logger.error(f"处理群聊消息时发生错误: {e}")

    def _is_duplicate_message(self, buffer: deque, new_msg: GroupMessage) -> bool:
        """检查消息是否已存在于缓冲区（防重复）"""
        # 如果新消息包含图片，则不视为重复，以确保图片总能被处理
        if new_msg.has_image:
            return False
            
        # 检查最近N条消息即可，避免性能问题
        recent_messages = list(buffer)[-self.config.duplicate_check_window_messages:] if buffer else []

        for existing_msg in recent_messages:
            # 重复判断条件：
            # 1. 相同发送者
            # 2. 相同文本内容
            # 3. 时间差在指定窗口内
            if (
                existing_msg.sender_id == new_msg.sender_id and
                existing_msg.text_content == new_msg.text_content and
                abs((new_msg.timestamp - existing_msg.timestamp).total_seconds()) < self.config.duplicate_check_time_seconds
            ):
                return True

        return False

    def _is_bot_message(self, event: AstrMessageEvent) -> bool:
        """检查是否是机器人自己发送的消息"""
        try:
            # 获取机器人自身ID
            bot_id = event.get_self_id()
            sender_id = event.get_sender_id()

            # 如果发送者ID等于机器人ID，则是机器人自己的消息
            return bool(bot_id and sender_id and str(sender_id) == str(bot_id))
        except (AttributeError, KeyError) as e:
            logger.debug(f"检查机器人消息时出错（可能是不支持的事件类型或数据结构）: {e}")
            return False

    def _classify_message(self, event: AstrMessageEvent) -> str:
        """分类消息类型"""

        # 🤖 首先检查是否是机器人消息
        if self._is_bot_message(event) and self.config.bot_replies_count > 0: # 逻辑上更合理的检查
            return ContextMessageType.BOT_REPLY

        # 检查是否包含图片
        if self._contains_image(event):
            return ContextMessageType.IMAGE_MESSAGE

        # 检查是否触发LLM
        if self._is_llm_triggered(event):
            # 附加一个唯一标识符，用于后续精确匹配
            setattr(event, '_context_enhancer_nonce', uuid.uuid4().hex)
            return ContextMessageType.LLM_TRIGGERED

        # 默认为普通聊天消息
        return ContextMessageType.NORMAL_CHAT

    def _contains_image(self, event: AstrMessageEvent) -> bool:
        """检查消息是否包含图片"""
        if not (event.message_obj and event.message_obj.message):
            return False

        for comp in event.message_obj.message:
            if isinstance(comp, Image):
                return True
        return False

    def _is_llm_triggered(self, event: AstrMessageEvent) -> bool:
        """判断消息是否会触发LLM回复（优化版）"""
        # 1. 检查唤醒状态 (最高效)
        if getattr(event, "is_wake", False) or getattr(
            event, "is_at_or_wake_command", False
        ):
            return True

        # 2. 检查@机器人
        if self._is_at_triggered(event):
            return True

        # 3. 检查命令前缀
        if self._is_keyword_triggered(event):
            return True

        return False

    def _is_at_triggered(self, event: AstrMessageEvent) -> bool:
        """检查消息是否通过@机器人触发"""
        if event.message_obj and event.message_obj.message:
            bot_id = event.get_self_id()
            for comp in event.message_obj.message:
                if isinstance(comp, At) and (
                    str(comp.qq) == str(bot_id) or comp.qq == "all"
                ):
                    return True
        return False

    def _is_keyword_triggered(self, event: AstrMessageEvent) -> bool:
        """检查消息是否通过命令前缀触发"""
        message_text = (event.message_str or "").lower().strip()
        if not message_text:
            return False

        return any(
            message_text.startswith(prefix)
            for prefix in self.config.command_prefixes
        )

    async def _generate_image_captions(self, group_msg: GroupMessage):
        """为图片生成智能描述，使用高级图片分析功能，支持独立的图片描述提供商"""
        try:
            if not group_msg.images:
                return

            # 检查是否启用图片描述
            if not self.config.enable_image_caption:
                # 如果禁用，使用简单占位符
                for i, img in enumerate(group_msg.images):
                    group_msg.image_captions.append(f"图片{i + 1}")
                return

            # 使用高级图片描述功能
            captions = []
            # 从统一配置中获取
            image_caption_provider_id = self.config.image_caption_provider_id
            image_caption_prompt = self.config.image_caption_prompt

            for i, img in enumerate(group_msg.images):
                try:
                    # 获取图片的URL或路径
                    image_data = getattr(img, "url", None) or getattr(img, "file", None)
                    if image_data and self.image_caption_utils is not None:
                        # 调用图片描述工具，传入特定的提供商ID和提示词
                        caption = await self.image_caption_utils.generate_image_caption(
                            image_data,
                            timeout=10,
                            provider_id=image_caption_provider_id
                            if image_caption_provider_id
                            else None,
                            custom_prompt=image_caption_prompt,
                        )
                        if caption:
                            # 直接存储纯净的描述文本
                            captions.append(caption)
                        else:
                            # 如果没有生成描述，可以添加一个默认占位符或空字符串
                            captions.append("图片")
                    else:
                        captions.append("图片")
                except Exception as e:
                    logger.debug(f"生成图片{i + 1}描述失败: {e}")
                    captions.append("图片")

            group_msg.image_captions = captions

        except Exception as e:
            logger.warning(f"生成图片描述时发生错误: {e}")
            # 降级到简单占位符
            for i, img in enumerate(group_msg.images):
                group_msg.image_captions.append(f"图片{i + 1}")

    @event_filter.on_llm_request(priority=100)
    async def on_llm_request(self, event: AstrMessageEvent, request: ProviderRequest):
        """LLM请求时提供简单直接的上下文增强"""
        try:
            # 简单检测：避免重复增强
            if request.prompt and "你正在浏览聊天软件" in request.prompt:
                logger.debug("检测到已增强的内容，跳过重复处理")
                return

            if not self.is_chat_enabled(event):
                logger.debug("上下文增强器：当前聊天未启用，跳过增强")
                return

            if event.get_message_type() != MessageType.GROUP_MESSAGE:
                return

            logger.debug("开始构建简单上下文...")

            # 标记当前消息为LLM触发类型
            await self._mark_current_as_llm_triggered(event)

            # 获取群聊历史
            group_id = (
                event.get_group_id()
                if hasattr(event, "get_group_id")
                else event.unified_msg_origin
            )
            buffer = self._get_group_buffer(group_id)

            if not buffer:
                logger.debug("没有群聊历史，跳过增强")
                return

            # 【重构】从 buffer 提取消息和图片
            extracted_data = self._extract_messages_for_context(buffer)

            # 格式化 prompt
            enhanced_prompt = self._format_prompt_from_messages(
                original_prompt=request.prompt,
                event=event,
                recent_chats=extracted_data["recent_chats"],
                bot_replies=extracted_data["bot_replies"],
            )

            if enhanced_prompt and enhanced_prompt != request.prompt:
                request.prompt = enhanced_prompt
                logger.debug(f"上下文增强完成，新prompt长度: {len(enhanced_prompt)}")

            # 根据配置截取最终的图片 URL 列表
            max_images = self.config.max_images_in_context
            final_image_urls = list(dict.fromkeys(extracted_data["image_urls"]))[
                -max_images:
            ]

            if final_image_urls:
                if not request.image_urls:
                    request.image_urls = []
                # 合并并去重
                request.image_urls = list(
                    dict.fromkeys(final_image_urls + request.image_urls)
                )
                logger.debug(f"上下文中新增了 {len(final_image_urls)} 张图片")

        except Exception as e:
            logger.error(f"上下文增强时发生错误: {e}")

    def _extract_messages_for_context(self, buffer: deque) -> dict:
        """从消息缓冲区中提取和筛选数据"""
        recent_chats = []
        bot_replies = []
        image_urls = []

        # 读取配置
        max_chats = self.config.recent_chats_count
        max_bot_replies = self.config.bot_replies_count

        # 从后向前遍历 buffer 来收集所需数量的消息
        for msg in reversed(buffer):
            if (
                len(recent_chats) < max_chats
                and msg.message_type
                in [
                    ContextMessageType.NORMAL_CHAT,
                    ContextMessageType.LLM_TRIGGERED,
                    ContextMessageType.IMAGE_MESSAGE,
                ]
            ):
                text_part = f"{msg.sender_name}: {msg.text_content}"
                caption_part = ""
                if msg.image_captions:
                    # 在格式化输出时动态添加前缀
                    caption_part = f" [图片: {'; '.join(msg.image_captions)}]"

                if msg.text_content or caption_part:
                    recent_chats.append(f"{text_part}{caption_part}")

                if msg.has_image:
                    for img in msg.images:
                        image_url = getattr(img, "url", None) or getattr(
                            img, "file", None
                        )
                        if image_url:
                            image_urls.append(image_url)

            elif (
                len(bot_replies) < max_bot_replies
                and msg.message_type == ContextMessageType.BOT_REPLY
            ):
                bot_replies.append(f"你回复了: {msg.text_content}")

            # 如果两类消息都已收集足够，则提前结束循环
            if len(recent_chats) >= max_chats and len(bot_replies) >= max_bot_replies:
                break
        
        # 反转列表以恢复正确的顺序
        recent_chats.reverse()
        bot_replies.reverse()
        image_urls.reverse()

        return {
            "recent_chats": recent_chats,
            "bot_replies": bot_replies,
            "image_urls": image_urls,
        }

    def _format_prompt_from_messages(
        self,
        original_prompt: str,
        event: AstrMessageEvent,
        recent_chats: list,
        bot_replies: list,
    ) -> str:
        """将提取出的数据格式化为最终的 prompt 字符串"""
        context_parts = [ContextConstants.PROMPT_HEADER]

        context_parts.extend(self._format_recent_chats_section(recent_chats))
        context_parts.extend(self._format_bot_replies_section(bot_replies))
        context_parts.append(self._format_situation_section(original_prompt, event))
        context_parts.append(ContextConstants.PROMPT_FOOTER)

        return "\n".join(part for part in context_parts if part)

    def _format_recent_chats_section(self, recent_chats: list) -> list:
        """格式化最近的聊天记录部分"""
        if not recent_chats:
            return []
        return [ContextConstants.RECENT_CHATS_HEADER] + recent_chats

    def _format_bot_replies_section(self, bot_replies: list) -> list:
        """格式化机器人回复部分"""
        if not bot_replies:
            return []
        return [ContextConstants.BOT_REPLIES_HEADER] + bot_replies

    def _format_situation_section(self, original_prompt: str, event: AstrMessageEvent) -> str:
        """格式化当前情景部分"""
        sender_id = event.get_sender_id()
        if sender_id:
            sender_name = event.get_sender_name() or "用户"
            return ContextConstants.USER_TRIGGER_TEMPLATE.format(
                sender_name=sender_name,
                sender_id=sender_id,
                original_prompt=original_prompt,
            )
        else:
            return ContextConstants.PROACTIVE_TRIGGER_TEMPLATE.format(
                original_prompt=original_prompt
            )

    @event_filter.on_llm_response(priority=100)
    async def on_llm_response(self, event: AstrMessageEvent, resp):
        """记录机器人的回复内容"""
        try:
            if event.get_message_type() == MessageType.GROUP_MESSAGE:
                group_id = (
                    event.get_group_id()
                    if hasattr(event, "get_group_id")
                    else event.unified_msg_origin
                )

                # 获取回复文本
                response_text = ""
                if hasattr(resp, "completion_text"):
                    response_text = resp.completion_text
                elif hasattr(resp, "text"):
                    response_text = resp.text
                else:
                    response_text = str(resp)

                # 创建机器人回复记录
                bot_reply = GroupMessage(
                    message_type=ContextMessageType.BOT_REPLY,
                    sender_id=event.get_self_id(),
                    sender_name=self.raw_config.get("name", "助手"),
                    group_id=group_id,
                    text_content=response_text[:1000]
                )

                buffer = self._get_group_buffer(group_id)
                buffer.append(bot_reply)

                logger.debug(f"记录机器人回复: {response_text[:50]}...")

        except Exception as e:
            logger.error(f"记录机器人回复时发生错误: {e}")

    def clear_context_cache(self, group_id: Optional[str] = None):
        """
        清空上下文缓存。
        如果提供了 group_id，则只清空该群组的缓存。
        否则，清空所有群组的缓存。
        """
        try:
            if group_id:
                # 清空特定群组的缓存
                if group_id in self.group_messages:
                    del self.group_messages[group_id]
                    logger.info(f"已清空群组 {group_id} 的内存上下文缓存。")
                if group_id in self.group_last_activity:
                    del self.group_last_activity[group_id]
                
                # 注意：文件缓存的原子性更新将在插件终止时统一处理
                # 这里可以选择立即重写文件，但为了性能和一致性，依赖于正常关闭流程
                logger.info(f"群组 {group_id} 的文件缓存将在下次保存时更新。")

            else:
                # 清空所有缓存
                logger.info(f"[诊断] 清空前 group_messages 包含 {len(self.group_messages)} 个群组。")
                self.group_messages.clear()
                self.group_last_activity.clear()
                logger.info("内存中的所有上下文缓存已清空。")

                if os.path.exists(self.cache_path):
                    os.remove(self.cache_path)
                    logger.info(f"持久化缓存文件 {self.cache_path} 已删除。")

        except Exception as e:
            logger.error(f"清空上下文缓存时发生错误: {e}")

    @event_filter.command("reset", "new", description="清空当前群聊的上下文缓存")
    async def handle_clear_context_command(self, event: AstrMessageEvent):
        """处理 reset 和 new 命令，清空特定群组的上下文缓存"""
        group_id = event.get_group_id()
        if group_id:
            logger.info(f"收到为群组 {group_id} 清空上下文的命令...")
            self.clear_context_cache(group_id=group_id)
        else:
            logger.warning("无法获取 group_id，无法执行定向清空操作。")

    async def _mark_current_as_llm_triggered(self, event: AstrMessageEvent):
        """将当前消息标记为LLM触发类型（增强版）"""
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return

        group_id = event.get_group_id() or event.unified_msg_origin
        buffer = self._get_group_buffer(group_id)
        
        # 定义一个较小的、固定的搜索窗口以优化性能
        search_window = list(buffer)[-20:]

        # 优先使用消息ID进行精确匹配
        msg_id = getattr(event, 'id', None) or getattr(event.message_obj, 'id', None)
        if msg_id:
            for msg in reversed(search_window):
                if getattr(msg, 'id', None) == msg_id:
                    msg.message_type = ContextMessageType.LLM_TRIGGERED
                    logger.debug(f"通过消息ID标记为LLM触发: {msg.text_content[:50]}...")
                    return

        # 其次，使用 nonce 进行精确匹配
        nonce = getattr(event, '_context_enhancer_nonce', None)
        if nonce:
            for msg in reversed(search_window):
                if msg.nonce == nonce:
                    msg.message_type = ContextMessageType.LLM_TRIGGERED
                    logger.debug(f"通过 nonce 标记为LLM触发: {msg.text_content[:50]}...")
                    return
        
        # 如果两种精确匹配都失败，则记录并放弃
        logger.warning(
            "无法通过消息ID或nonce找到要标记的消息，放弃标记。这可能发生在消息处理延迟或状态不一致时。"
        )
