use crate::core::{
    aiserver::v1::{StreamChatResponse, WebReference},
    error::{ChatError, StreamError},
};
use bytes::{Buf, BytesMut};
use flate2::read::GzDecoder;
use prost::Message;
use std::io::Read;
use std::time::Instant;

pub trait InstantExt {
    fn duration_as_secs_f32(&mut self) -> f32;
}

impl InstantExt for Instant {
    #[inline]
    fn duration_as_secs_f32(&mut self) -> f32 {
        let now = Instant::now();
        let duration = now.duration_since(*self);
        *self = now;
        duration.as_secs_f32()
    }
}

/// 解压gzip数据
#[inline]
fn decompress_gzip(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 3
        || unsafe { *data.get_unchecked(0) } != 0x1f
        || unsafe { *data.get_unchecked(1) } != 0x8b
        || unsafe { *data.get_unchecked(2) } != 0x08
    {
        return None;
    }

    let mut decoder = GzDecoder::new(data);
    let mut decompressed = Vec::new();

    match decoder.read_to_end(&mut decompressed) {
        Ok(_) => Some(decompressed),
        Err(_) => {
            // println!("gzip解压失败: {}", e);
            None
        }
    }
}

#[derive(PartialEq, Clone)]
pub enum StreamMessage {
    // 调试
    Debug(String),
    // 网络引用
    WebReference(Vec<WebReference>),
    // 内容开始标志
    #[cfg(test)]
    ContentStart,
    // 消息内容
    Content(String),
    // 额度消耗
    Usage(String),
    // 流结束标志
    StreamEnd,
}

impl StreamMessage {
    #[inline]
    fn convert_web_ref_to_content(self) -> Self {
        match self {
            StreamMessage::WebReference(refs) => {
                if refs.is_empty() {
                    return StreamMessage::Content(String::new());
                }

                let mut result = String::from("WebReferences:\n");
                for (i, web_ref) in refs.iter().enumerate() {
                    result.push_str(&format!(
                        "{}. [{}]({})<{}>\n",
                        i + 1,
                        web_ref.title,
                        web_ref.url,
                        web_ref.chunk
                    ));
                }
                result.push('\n');
                StreamMessage::Content(result)
            }
            other => other,
        }
    }
}

pub struct StreamDecoder {
    // 主要数据缓冲区
    buffer: BytesMut,
    // 结果相关 (24字节 + 48字节)
    first_result: Option<Vec<StreamMessage>>,
    content_delays: Option<(String, Vec<(u32, f32)>)>,
    // 计数器和时间 (8字节 + 8字节)
    empty_stream_count: usize,
    last_content_time: Instant,
    // 状态标志 (1字节 + 1字节 + 1字节)
    first_result_ready: bool,
    first_result_taken: bool,
    has_seen_content: bool,
}

impl StreamDecoder {
    pub fn new() -> Self {
        Self {
            buffer: BytesMut::new(),
            first_result: None,
            content_delays: None,
            empty_stream_count: 0,
            last_content_time: Instant::now(),
            first_result_ready: false,
            first_result_taken: false,
            has_seen_content: false,
        }
    }

    #[inline]
    pub fn get_empty_stream_count(&self) -> usize {
        self.empty_stream_count
    }

    #[inline]
    pub fn reset_empty_stream_count(&mut self) {
        if self.empty_stream_count > 0 {
            crate::debug_println!(
                "重置连续空流计数，之前的计数为: {}",
                self.empty_stream_count
            );
            self.empty_stream_count = 0;
        }
    }

    #[inline]
    pub fn take_first_result(&mut self) -> Option<Vec<StreamMessage>> {
        if !self.buffer.is_empty() {
            return None;
        }
        if self.first_result.is_some() {
            self.first_result_taken = true;
        }
        self.first_result.take()
    }

    #[cfg(test)]
    fn is_incomplete(&self) -> bool {
        !self.buffer.is_empty()
    }

    #[inline]
    pub fn is_first_result_ready(&self) -> bool {
        self.first_result_ready
    }

    #[inline]
    pub fn take_content_delays(&mut self) -> Option<(String, Vec<(u32, f32)>)> {
        std::mem::take(&mut self.content_delays)
    }

    #[inline]
    pub fn no_first_cache(mut self) -> Self {
        self.first_result_ready = true;
        self.first_result_taken = true;
        self
    }

    pub fn decode(
        &mut self,
        data: &[u8],
        convert_web_ref: bool,
    ) -> Result<Vec<StreamMessage>, StreamError> {
        if !data.is_empty() {
            self.reset_empty_stream_count();
        }

        self.buffer.extend_from_slice(data);

        if self.buffer.len() < 5 {
            if self.buffer.is_empty() {
                self.empty_stream_count += 1;

                return Err(StreamError::EmptyStream);
            }
            crate::debug_println!("数据长度小于5字节，当前数据: {}", hex::encode(&self.buffer));
            return Err(StreamError::DataLengthLessThan5);
        }

        self.reset_empty_stream_count();

        let reserve = {
            let mut offset = 0;
            let mut count = 0;
            while offset + 5 <= self.buffer.len() {
                let msg_len: usize;

                // SAFETY: The loop condition `offset + 5 <= self.buffer.len()` guarantees
                // that indices `offset` through `offset + 4` are within bounds.
                unsafe {
                    msg_len = u32::from_be_bytes([
                        *self.buffer.get_unchecked(offset + 1),
                        *self.buffer.get_unchecked(offset + 2),
                        *self.buffer.get_unchecked(offset + 3),
                        *self.buffer.get_unchecked(offset + 4),
                    ]) as usize;
                }

                if msg_len == 0 {
                    offset += 5;
                    continue;
                }

                if offset + 5 + msg_len > self.buffer.len() {
                    break;
                }

                offset += 5 + msg_len;
                count += 1;
            }
            count
        };

        if let Some(content_delays) = self.content_delays.as_mut() {
            content_delays.0.reserve(reserve);
            content_delays.1.reserve(reserve);
        } else {
            self.content_delays =
                Some((String::with_capacity(reserve), Vec::with_capacity(reserve)));
        }

        let mut messages = Vec::with_capacity(reserve);
        let mut offset = 0;

        while offset + 5 <= self.buffer.len() {
            let msg_type: u8;
            let msg_len: usize;

            // SAFETY: The loop condition `offset + 5 <= self.buffer.len()` guarantees
            // that indices `offset` through `offset + 4` are within bounds.
            unsafe {
                msg_type = *self.buffer.get_unchecked(offset);
                msg_len = u32::from_be_bytes([
                    *self.buffer.get_unchecked(offset + 1),
                    *self.buffer.get_unchecked(offset + 2),
                    *self.buffer.get_unchecked(offset + 3),
                    *self.buffer.get_unchecked(offset + 4),
                ]) as usize;
            }

            if msg_len == 0 {
                offset += 5;
                #[cfg(test)]
                messages.push(StreamMessage::ContentStart);
                continue;
            }

            if offset + 5 + msg_len > self.buffer.len() {
                break;
            }

            let msg_data = &self.buffer[offset + 5..offset + 5 + msg_len];

            if let Some(msg) = self.process_message(msg_type, msg_data)? {
                if let StreamMessage::Content(content) = &msg {
                    self.has_seen_content = true;
                    let delay = self.last_content_time.duration_as_secs_f32();
                    if let Some(content_delays) = self.content_delays.as_mut() {
                        content_delays.0.push_str(content);
                        content_delays
                            .1
                            .push((content.chars().count() as u32, delay));
                    }
                }
                if convert_web_ref {
                    messages.push(msg.convert_web_ref_to_content());
                } else {
                    messages.push(msg);
                }
            }

            offset += 5 + msg_len;
        }

        self.buffer.advance(offset);

        if !self.first_result_taken && !messages.is_empty() {
            if self.first_result.is_none() {
                self.first_result = Some(std::mem::take(&mut messages));
            } else if !self.first_result_ready {
                if let Some(first_result) = &mut self.first_result {
                    first_result.append(&mut messages);
                }
            }
        }
        if !self.first_result_ready {
            self.first_result_ready = self.first_result.is_some()
                && self.buffer.is_empty()
                && !self.first_result_taken
                && self.has_seen_content;
        }
        Ok(messages)
    }

    #[inline]
    fn process_message(
        &self,
        msg_type: u8,
        msg_data: &[u8],
    ) -> Result<Option<StreamMessage>, StreamError> {
        match msg_type {
            0 => self.handle_text_message(msg_data),
            1 => self.handle_gzip_message(msg_data),
            2 => self.handle_json_message(msg_data),
            3 => self.handle_gzip_json_message(msg_data),
            t => {
                eprintln!("收到未知消息类型: {t}，请尝试联系开发者以获取支持");
                crate::debug_println!("消息类型: {t}，消息内容: {}", hex::encode(msg_data));
                Ok(None)
            }
        }
    }

    #[inline]
    fn handle_text_message(&self, msg_data: &[u8]) -> Result<Option<StreamMessage>, StreamError> {
        if let Ok(response) = StreamChatResponse::decode(msg_data) {
            // crate::debug_println!("[text] StreamChatResponse [hex: {}]: {:#?}", hex::encode(msg_data), response);
            if !response.text.is_empty() {
                Ok(Some(StreamMessage::Content(response.text)))
            } else if let Some(filled_prompt) = response.filled_prompt {
                Ok(Some(StreamMessage::Debug(filled_prompt)))
            } else if let Some(web_citation) = response.web_citation {
                Ok(Some(StreamMessage::WebReference(web_citation.references)))
            } else if let Some(usage_uuid) = response.usage_uuid {
                Ok(Some(StreamMessage::Usage(usage_uuid)))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    #[inline]
    fn handle_gzip_message(&self, msg_data: &[u8]) -> Result<Option<StreamMessage>, StreamError> {
        if let Some(text) = decompress_gzip(msg_data) {
            if let Ok(response) = StreamChatResponse::decode(&text[..]) {
                // crate::debug_println!("[gzip] StreamChatResponse [hex: {}]: {:#?}", hex::encode(msg_data), response);
                if !response.text.is_empty() {
                    Ok(Some(StreamMessage::Content(response.text)))
                } else if let Some(filled_prompt) = response.filled_prompt {
                    Ok(Some(StreamMessage::Debug(filled_prompt)))
                } else if let Some(web_citation) = response.web_citation {
                    Ok(Some(StreamMessage::WebReference(web_citation.references)))
                } else if let Some(usage_uuid) = response.usage_uuid {
                    Ok(Some(StreamMessage::Usage(usage_uuid)))
                } else {
                    Ok(None)
                }
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    #[inline]
    fn handle_json_message(&self, msg_data: &[u8]) -> Result<Option<StreamMessage>, StreamError> {
        if msg_data.len() == 2 {
            return Ok(Some(StreamMessage::StreamEnd));
        }
        if let Ok(text) = String::from_utf8(msg_data.to_vec()) {
            // crate::debug_println!("[text] JSON消息 [hex: {}]: {}", hex::encode(msg_data), text);
            if let Ok(error) = serde_json::from_str::<ChatError>(&text) {
                return Err(StreamError::ChatError(error));
            }
        }
        Ok(None)
    }

    #[inline]
    fn handle_gzip_json_message(
        &self,
        msg_data: &[u8],
    ) -> Result<Option<StreamMessage>, StreamError> {
        if let Some(text) = decompress_gzip(msg_data) {
            if text.len() == 2 {
                return Ok(Some(StreamMessage::StreamEnd));
            }
            if let Ok(text) = String::from_utf8(text) {
                // crate::debug_println!("[gzip] JSON消息 [hex: {}]: {}", hex::encode(msg_data), text);
                if let Ok(error) = serde_json::from_str::<ChatError>(&text) {
                    return Err(StreamError::ChatError(error));
                }
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_chunk() {
        // 使用include_str!加载测试数据文件
        let stream_data = include_str!("../../../tests/data/stream_data.txt");

        // 将整个字符串按每两个字符分割成字节
        let bytes: Vec<u8> = stream_data
            .as_bytes()
            .chunks(2)
            .map(|chunk| {
                let hex_str = std::str::from_utf8(chunk).unwrap();
                u8::from_str_radix(hex_str, 16).unwrap()
            })
            .collect();

        // 创建解码器
        let mut decoder = StreamDecoder::new().no_first_cache();

        match decoder.decode(&bytes, false) {
            Ok(messages) => {
                for message in messages {
                    match message {
                        StreamMessage::StreamEnd => {
                            println!("流结束");
                            break;
                        }
                        StreamMessage::Usage(msg) => {
                            println!("额度uuid: {msg}");
                        }
                        StreamMessage::Content(msg) => {
                            println!("消息内容: {msg}");
                        }
                        StreamMessage::WebReference(refs) => {
                            println!("网页引用:");
                            for (i, web_ref) in refs.iter().enumerate() {
                                println!(
                                    "{}. {} - {} - {}",
                                    i, web_ref.url, web_ref.title, web_ref.chunk
                                );
                            }
                        }
                        StreamMessage::Debug(prompt) => {
                            println!("调试信息: {prompt}");
                        }
                        StreamMessage::ContentStart => {
                            println!("流开始");
                        }
                    }
                }
            }
            Err(e) => {
                println!("解析错误: {e}");
            }
        }
        if decoder.is_incomplete() {
            println!("数据不完整");
        }
    }

    #[test]
    fn test_multiple_chunks() {
        // 使用include_str!加载测试数据文件
        let stream_data = include_str!("../../../tests/data/stream_data.txt");

        // 将整个字符串按每两个字符分割成字节
        let bytes: Vec<u8> = stream_data
            .as_bytes()
            .chunks(2)
            .map(|chunk| {
                let hex_str = std::str::from_utf8(chunk).unwrap();
                u8::from_str_radix(hex_str, 16).unwrap()
            })
            .collect();

        // 创建解码器
        let mut decoder = StreamDecoder::new().no_first_cache();

        // 辅助函数：找到下一个消息边界
        fn find_next_message_boundary(bytes: &[u8]) -> usize {
            if bytes.len() < 5 {
                return bytes.len();
            }
            let msg_len = u32::from_be_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]) as usize;
            5 + msg_len
        }

        // 辅助函数：将字节转换为hex字符串
        fn bytes_to_hex(bytes: &[u8]) -> String {
            bytes
                .iter()
                .map(|b| format!("{:02X}", b))
                .collect::<Vec<String>>()
                .join("")
        }

        // 多次解析数据
        let mut offset = 0;
        let mut should_break = false;

        while offset < bytes.len() {
            let remaining_bytes = &bytes[offset..];
            let msg_boundary = find_next_message_boundary(remaining_bytes);
            let current_msg_bytes = &remaining_bytes[..msg_boundary];
            let hex_str = bytes_to_hex(current_msg_bytes);

            match decoder.decode(current_msg_bytes, false) {
                Ok(messages) => {
                    for message in messages {
                        match message {
                            StreamMessage::StreamEnd => {
                                println!("流结束 [hex: {hex_str}]");
                                should_break = true;
                                break;
                            }
                            StreamMessage::Usage(msg) => {
                                println!("额度uuid: {msg}");
                            }
                            StreamMessage::Content(msg) => {
                                println!("消息内容 [hex: {hex_str}]: {msg}");
                            }
                            StreamMessage::WebReference(refs) => {
                                println!("网页引用 [hex: {hex_str}]:");
                                for (i, web_ref) in refs.iter().enumerate() {
                                    println!(
                                        "{}. {} - {} - {}",
                                        i, web_ref.url, web_ref.title, web_ref.chunk
                                    );
                                }
                            }
                            StreamMessage::Debug(prompt) => {
                                println!("调试信息 [hex: {hex_str}]: {prompt}");
                            }
                            StreamMessage::ContentStart => {
                                println!("流开始 [hex: {hex_str}]");
                            }
                        }
                    }
                    if should_break {
                        break;
                    }
                    if decoder.is_incomplete() {
                        println!("数据不完整 [hex: {}]", hex_str);
                        break;
                    }
                    offset += msg_boundary;
                }
                Err(e) => {
                    println!("解析错误 [hex: {}]: {}", hex_str, e);
                    break;
                }
            }
        }
    }
}
