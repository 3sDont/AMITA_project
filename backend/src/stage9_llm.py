"""
Stage 9: LLM Analysis (IMPROVED VERSION)
Generate summary, extract tasks, insights

✅ IMPROVEMENTS:
    - Better prompt engineering với system prompts
    - Chunking strategy cho meeting dài
    - Few-shot examples để giảm hallucination
    - Output validation và error handling
    - Context window optimization (up to 8000 tokens)
"""
from typing import Dict, List, Optional
import json
import re

try:
    from ollama import chat, list as ollama_list
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class LLMAnalyzer:
    """
    Enhanced LLM Analyzer - Phân tích meeting với quality improvements
    
    New Features:
        - System prompts cho từng task
        - Smart chunking (8000 char context window)
        - Few-shot examples
        - Output validation
        - Deduplication
        - Better error handling
    """
    
    def __init__(self, model: str = "llama3.2:3b", max_context_length: int = 8000):
        """
        Initialize LLM Analyzer
        
        Args:
            model: Ollama model name
            max_context_length: Maximum context length (characters)
        """
        self.model = model
        self.max_context_length = max_context_length
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if Ollama and model are available"""
        if not OLLAMA_AVAILABLE:
            return False
        
        try:
            models = ollama_list()
            model_names = [m.model for m in models.models]
            #model_names = [m['name'] for m in models.get('models', [])]
            model_exists = any(
                self.model in name or self.model.split(':')[0] in name 
                for name in model_names
            )
            return model_exists
        except Exception:
            return False
    
    def analyze(
        self,
        segments: List[Dict],
        speakers: Dict,
        metadata: Dict
    ) -> Dict:
        """
        Main analysis function
        
        Returns:
            Dict: {
                summary: str,
                tasks: List[Dict],
                insights: List[str]
            }
        """
        if not self.available:
            print(f"   ⚠️  Model '{self.model}' not available")
            print(f"   💡 To use LLM features:")
            print(f"      1. Install Ollama: https://ollama.ai")
            print(f"      2. Pull model: ollama pull {self.model}")
            return {
                'summary': "[LLM not available - please install Ollama and pull model]",
                'tasks': [],
                'insights': []
            }
        
        print(f"   🤖 Using model: {self.model}")
        
        # Build context
        context = self._build_context(segments, speakers, metadata)
        
        # Generate summary
        print(f"   📝 Generating summary...")
        summary = self._generate_summary(context)
        
        # Extract tasks
        print(f"   ✅ Extracting tasks...")
        tasks = self._extract_tasks(context)
        
        # Extract insights
        print(f"   💡 Extracting insights...")
        insights = self._extract_insights(context)
        
        return {
            'summary': summary,
            'tasks': tasks,
            'insights': insights
        }
    
    def _build_context(
        self, 
        segments: List[Dict], 
        speakers: Dict, 
        metadata: Dict,
        max_length: Optional[int] = None
    ) -> str:
        """Build context string với smart formatting"""
        if max_length is None:
            max_length = self.max_context_length
        
        lines = []
        
        # Header
        lines.append("=== MEETING INFORMATION ===")
        lines.append(f"Meeting ID: {metadata.get('meeting_id', 'Unknown')}")
        lines.append(f"Total Segments: {len(segments)}")
        
        if segments:
            total_duration = segments[-1].get('end_time', 0)
            minutes = int(total_duration // 60)
            lines.append(f"Duration: {minutes} minutes")
        
        if speakers:
            lines.append(f"Participants: {len(speakers)} speakers")
        
        lines.append("\n=== TRANSCRIPT ===")
        
        # Build transcript
        current_length = len("\n".join(lines))
        
        for seg in segments:
            speaker = seg.get('speaker_display', seg['speaker'])
            text = seg.get('text', '')
            time_str = seg.get('time_str', '')
            
            line = f"[{time_str}] {speaker}: {text}"
            
            if current_length + len(line) + 1 > max_length:
                lines.append("\n[... transcript truncated ...]")
                break
            
            lines.append(line)
            current_length += len(line) + 1
        
        return "\n".join(lines)
    
    def _chunk_context(self, context: str, chunk_size: int = 6000, overlap: int = 500) -> List[str]:
        """Split context into overlapping chunks"""
        if len(context) <= chunk_size:
            return [context]
        
        chunks = []
        start = 0
        
        while start < len(context):
            end = start + chunk_size
            
            # Find last newline to avoid cutting mid-sentence
            if end < len(context):
                last_newline = context.rfind('\n', start, end)
                if last_newline > start:
                    end = last_newline
            
            chunk = context[start:end]
            chunks.append(chunk)
            
            start = end - overlap
            if start >= len(context):
                break
        
        return chunks
    
    def _generate_summary(self, context: str) -> str:
        """Generate summary với improved prompts"""
        system_prompt = """Vai trò: Bạn là một Thư ký điều hành và Chuyên gia phân tích dữ liệu hội thoại. Nhiệm vụ của bạn là trích xuất "giá trị cốt lõi" từ bản ghi chép cuộc họp (transcript) dưới đây.
QUY TẮC XỬ LÝ DỮ LIỆU:
Trung thực: Chỉ sử dụng thông tin có trong transcript. Tuyệt đối không suy diễn ý định của người nói.
Loại bỏ nhiễu: Bỏ qua các lời chào hỏi, chuyện phiếm, hoặc các thảo luận ngoài lề không đi đến kết quả.
Độ dài: Tối ưu trong khoảng 150-250 từ để đảm bảo đủ ý nhưng vẫn súc tích.
Đối tượng: Viết cho cấp quản lý đọc để nắm bắt tình hình trong 1 phút.
CẤU TRÚC ĐẦU RA (FORMAT):

🎯 Mục tiêu chính: [Tóm tắt lý do cuộc họp diễn ra trong 1 câu]

📋 Nội dung thảo luận trọng tâm:
Điểm 1: [Vấn đề thảo luận + Ý kiến chính/Giải pháp đưa ra]
Điểm 2: [Vấn đề thảo luận + Ý kiến chính/Giải pháp đưa ra]
(Tối đa 3-4 gạch đầu dòng tùy vào độ dài cuộc họp)

✅ Kết luận & Quyết định:
[Ghi rõ các quyết định đã được thống nhất hoặc sự đồng thuận cuối cùng]"""

        chunks = self._chunk_context(context, chunk_size=6000)
        
        if len(chunks) == 1:
            user_prompt = f"""Dựa vào transcript cuộc họp, hãy tóm tắt theo format đã cho:

{chunks[0]}

TÓM TẮT:"""
            
            response = chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": 0.3,  # Deterministic output
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            )
            
            return response["message"]["content"].strip()
        
        else:
            # Long meeting - chunked processing
            print(f"      💡 Long meeting: {len(chunks)} chunks")
            
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                print(f"         Processing chunk {i+1}/{len(chunks)}...")
                
                user_prompt = f"""Tóm tắt phần này (50-100 từ):

{chunk}

TÓM TẮT:"""
                
                response = chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    options={
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1
                    }
                )
                
                chunk_summaries.append(response["message"]["content"].strip())
            
            # Combine
            combined = "\n\n".join([f"Phần {i+1}: {s}" for i, s in enumerate(chunk_summaries)])
            
            final_prompt = f"""Kết hợp các phần tóm tắt thành TÓM TẮT TỔNG HỢP:

{combined}

TÓM TẮT TỔNG HỢP:"""
            
            response = chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": final_prompt}
                ],
                options={
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            )
            
            return response["message"]["content"].strip()
    
    def _extract_tasks(self, context: str) -> List[Dict]:
        """Extract tasks với validation"""
        system_prompt = """
Vai trò: Bạn là một Trợ lý Cuộc Họp chuyên nghiệp, có khả năng phân tích hội thoại và chuyển hóa các thảo luận thành danh sách công việc thực thi (Actionable Tasks).

NHIỆM VỤ: TRÍCH XUẤT TẤT CẢ các TASK/CÔNG VIỆC từ transcript cuộc họp và trả về định dạng JSON đã chuẩn hóa.

QUY TẮC NGHIÊM NGẶT:
1. CHỈ trích xuất tasks được ĐỀ CẬP RÕ RÀNG trong transcript
2. KHÔNG tự suy luận, KHÔNG gán thêm trách nhiệm, KHÔNG suy diễn mục tiêu ẩn
3. TASK hợp lệ PHẢI có ít nhất 1 trong các dấu hiệu: Có người thực hiện, Có deadline, Có động từ hành động
4. KHÔNG phải TASK: Thảo luận chung, nhận xét, chia sẻ thông tin; chỉ trích, phản hồi, tóm tắt, cảm ơn, hỏi đáp; 
5. Có người thực hiện → ghi tên
6. Có deadline → ghi cụ thể
7. Không có task nào → trả về []

DẤU HIỆU TASK:
- Từ khóa: "sẽ", "phải", "cần", "được giao", "nhớ", "bắt buộc", "yêu cầu", "deadline", "trước ngày", "hoàn thành vào"
- Động từ: "chuẩn bị", "liên hệ", "gửi", "kiểm tra", "hoàn thành"
- Có deadline hoặc người thực hiện


QUAN TRỌNG - FIELD "how_to":
- Nếu cuộc họp ĐÃ NÓI cách thực hiện → Ghi lại chính xác
- Nếu cuộc họp CHƯA NÓI → Dựa vào context và kinh nghiệm, đề xuất 1-2 bước cụ thể
- LUÔN PHẢI có nội dung hữu ích, KHÔNG để trống hoặc chung chung
- Format: Viết liền trên 1 dòng, dùng dấu phẩy ngăn cách các bước

QUAN TRỌNG - FORMAT JSON (BẮT BUỘC - KHÔNG VI PHẠM):
- Output PHẢI là JSON ARRAY gồm các OBJECT
- KHÔNG sử dụng dấu xuống dòng trong bất kỳ giá trị string nào
- Tất cả nội dung phải viết liền trên 1 dòng
- Nếu cần nhiều ý, dùng dấu phẩy (,) hoặc chấm phẩy (;) để ngăn cách
- Đảm bảo đóng dấu ngoặc kép đúng cách

OUTPUT FORMAT (JSON ARRAY - KHÔNG XUỐNG DÒNG):
[
  {
    "task": "Công việc ngắn gọn (5-15 từ)",
    "assigned_to": "Tên người hoặc null",
    "deadline": "Thời hạn hoặc null",
    "priority": "high/medium/low",
    "how_to": "Hướng dẫn CỤ THỂ 1-2 câu, viết liền không xuống dòng, dùng dấu phẩy hoặc chấm phẩy ngăn cách các bước"
  }
]"""

#         few_shot = """
# VÍ DỤ:

# Input: "Anh Minh sẽ chuẩn bị báo cáo Q4 trước ngày 15. Em Lan liên hệ team IT setup server."

# Output:
# [
#   {
#     "task": "Chuẩn bị báo cáo Q4",
#     "assigned_to": "Anh Minh",
#     "deadline": "Trước ngày 15",
#     "priority": "high",
#     "how_to": "Thu thập số liệu doanh thu, chi phí, lợi nhuận Q4; Tạo file báo cáo theo template công ty; Gửi approval"
#   },
#   {
#     "task": "Liên hệ team IT setup server",
#     "assigned_to": "Em Lan",
#     "deadline": null,
#     "priority": "medium",
#     "how_to": "Soạn email mô tả yêu cầu cấu hình server (RAM, CPU, storage), gửi đến it-support@company.com hoặc liên hệ trực tiếp qua Slack"
#   }
# ]

# CHÚ Ý: 
# - Field "how_to" phải CỤ THỂ và KHẢ THI
# - Tất cả string phải viết liền, KHÔNG xuống dòng
# - Dùng dấu chấm phẩy (;) hoặc dấu phẩy (,) để ngăn cách các bước
# """

        chunks = self._chunk_context(context, chunk_size=6000)
        all_tasks = []
        
        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                print(f"         Chunk {i+1}/{len(chunks)}...")
            
            user_prompt = f"""

Trích xuất tasks từ transcript:

{chunk}

JSON OUTPUT:"""
            
            try:
                response = chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    options={
                        "temperature": 0.0,  # Critical for consistent JSON output
                        "top_p": 0.9,
                        "repeat_penalty": 1.1
                    }
                )
                
                content = response["message"]["content"].strip()
                print(f"            📄 LLM response length: {len(content)} chars")
                
                # Clean up content - remove markdown code blocks
                content = re.sub(r'```json\s*', '', content)
                content = re.sub(r'```\s*', '', content)
                
                # Extract JSON array (first match only)
                match = re.search(r'\[.*?\]', content, re.DOTALL)
                if match:
                    json_str = match.group()
                    print(f"            ✓ Found JSON array")
                    
                    # ✅ SANITIZE JSON - Fix common LLM JSON errors
                    json_str = self._sanitize_json(json_str)
                    
                    # DEBUG: Log raw JSON
                    print(f"            🔍 Raw JSON (first 500 chars):")
                    print(f"            {json_str[:500]}")
                    
                    try:
                        tasks = json.loads(json_str)
                        if isinstance(tasks, list):
                            print(f"            ✓ Parsed {len(tasks)} raw tasks")
                            validated = self._validate_tasks(tasks)
                            print(f"            ✓ Validated {len(validated)} tasks")
                            all_tasks.extend(validated)
                        else:
                            print(f"         ⚠️  Chunk {i+1}: Expected list, got {type(tasks)}")
                    except json.JSONDecodeError as je:
                        print(f"         ⚠️  Chunk {i+1}: JSON decode error - {je}")
                        print(f"            🔍 Error location: line {je.lineno}, col {je.colno}")
                        
                        # Show context around error
                        if hasattr(je, 'pos'):
                            error_pos = je.pos
                            start = max(0, error_pos - 100)
                            end = min(len(json_str), error_pos + 100)
                            print(f"            Context: ...{json_str[start:end]}...")
                        
                        # ✅ FALLBACK: Try aggressive fixes
                        print(f"            🔧 Attempting aggressive JSON repair...")
                        fixed_tasks = self._fallback_parse_tasks(json_str)
                        if fixed_tasks:
                            print(f"            ✓ Recovered {len(fixed_tasks)} tasks via fallback")
                            all_tasks.extend(fixed_tasks)
                else:
                    print(f"         ⚠️  Chunk {i+1}: No JSON array found in response")
            
            except Exception as e:
                print(f"         ⚠️  Error in chunk {i+1}: {e}")
                continue
        
        # Deduplicate
        print(f"         📊 Total tasks before dedup: {len(all_tasks)}")
        unique_tasks = self._deduplicate_tasks(all_tasks)
        print(f"         📊 Unique tasks after dedup: {len(unique_tasks)}")
        return unique_tasks
    
    def _validate_tasks(self, tasks: List[Dict]) -> List[Dict]:
        """Validate task objects"""
        validated = []
        
        for task in tasks:
            if 'task' not in task or not task['task']:
                print(f"            ⚠️ Skipped: Missing 'task' field")
                continue
            
            # Skip generic tasks (less strict)
            generic = ['thảo luận chung', 'nói về chung', 'chia sẻ thông tin']
            task_lower = task['task'].lower()
            if any(kw in task_lower for kw in generic):
                print(f"            ⚠️ Skipped generic: {task['task'][:50]}")
                continue
            
            # Create validated task
            validated_task = {
                'task': task['task'].strip(),
                'assigned_to': task.get('assigned_to'),
                'deadline': task.get('deadline'),
                'priority': task.get('priority', 'medium'),
                'how_to': task.get('how_to', 'Chi tiết thực hiện sẽ được bổ sung.')
            }
            
            if validated_task['priority'] not in ['high', 'medium', 'low']:
                validated_task['priority'] = 'medium'
            
            validated.append(validated_task)
        
        return validated
    
    def _deduplicate_tasks(self, tasks: List[Dict]) -> List[Dict]:
        """Remove duplicate tasks"""
        if len(tasks) <= 1:
            return tasks
        
        unique = []
        seen = set()
        
        for task in tasks:
            task_key = task['task'].lower().strip()
            
            if task_key in seen:
                continue
            
            # Check similarity
            is_dup = False
            for existing in unique:
                existing_key = existing['task'].lower().strip()
                if task_key in existing_key or existing_key in task_key:
                    is_dup = True
                    break
            
            if not is_dup:
                unique.append(task)
                seen.add(task_key)
        
        return unique
    
    def spell_check_segments(self, segments: List[Dict], batch_size: int = 5) -> List[Dict]:
        """Chỉnh sửa chính tả và ngữ pháp cho các segments
        
        Args:
            segments: Danh sách segments cần chỉnh sửa
            batch_size: Số segments xử lý cùng lúc để tối ưu hóa
            
        Returns:
            List[Dict]: Segments với text đã được chỉnh sửa
        """
        if not self.available:
            print(f"   ⚠️  LLM not available - skipping spell check")
            return segments
        
        print(f"   📝 Starting spell check for {len(segments)} segments...")
        
        system_prompt = """Bạn là chuyên gia chỉnh sửa văn bản tiếng Việt.

NHIỆM VỤ: Sửa lỗi chính tả, ngữ pháp và cải thiện độ tự nhiên của văn bản

QUY TẮC BẮT BUỘC:
- CHỈ sửa lỗi chính tả, ngữ pháp, dấu câu
- GIỮ NGUYÊN ý nghĩa và nội dung gốc 100%
- KHÔNG thêm, bớt hoặc thay đổi thông tin
- KHÔNG diễn giải hoặc tóm tắt
- Giữ nguyên tên riêng, số liệu, thuật ngữ
- Nếu văn bản đã đúng → trả về NGUYÊN VĂN

LỖI THƯỜNG GẶP CẦN SỬA:
- Lỗi nhận dạng từ ASR (ví dụ: "ơn chào" → "xin chào")
- Thiếu dấu hoặc dấu sai
- Sai ngữ pháp cơ bản
- Từ viết liền hoặc tách nhầm

OUTPUT: CHỈ trả về văn bản đã sửa, KHÔNG giải thích."""
        
        corrected_segments = []
        total_corrected = 0
        
        # Process in batches for efficiency
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i+batch_size]
            
            # Build batch prompt
            batch_texts = []
            for idx, seg in enumerate(batch):
                text = seg.get('text', '').strip()
                if text:
                    batch_texts.append(f"{idx+1}. {text}")
            
            if not batch_texts:
                corrected_segments.extend(batch)
                continue
            
            user_prompt = f"""Sửa lỗi chính tả và ngữ pháp cho các câu sau:

{chr(10).join(batch_texts)}

VĂN BẢN ĐÃ SỬA (giữ nguyên số thứ tự):"""
            
            try:
                response = chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                
                corrected_text = response["message"]["content"].strip()
                
                # Parse response
                corrected_lines = {}
                for line in corrected_text.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Try to match pattern: "1. text" or "1) text"
                    match = re.match(r'^(\d+)[.):]\s*(.+)$', line)
                    if match:
                        num = int(match.group(1))
                        text = match.group(2).strip()
                        corrected_lines[num] = text
                
                # Apply corrections
                for idx, seg in enumerate(batch):
                    new_seg = seg.copy()
                    line_num = idx + 1
                    
                    if line_num in corrected_lines:
                        original = seg.get('text', '').strip()
                        corrected = corrected_lines[line_num]
                        
                        # Only update if there's a meaningful change
                        if corrected and corrected != original:
                            new_seg['text'] = corrected
                            new_seg['text_original'] = original  # Keep original for reference
                            total_corrected += 1
                    
                    corrected_segments.append(new_seg)
                
                if (i // batch_size + 1) % 10 == 0:
                    print(f"      Progress: {i+batch_size}/{len(segments)} segments processed...")
            
            except Exception as e:
                print(f"      ⚠️  Error in batch {i//batch_size + 1}: {e}")
                # Keep original segments on error
                corrected_segments.extend(batch)
        
        print(f"   ✅ Spell check complete: {total_corrected}/{len(segments)} segments corrected")
        return corrected_segments
    
    def _extract_insights(self, context: str) -> List[str]:
        """Extract key insights"""
        system_prompt = """Bạn là chuyên gia phân tích cuộc họp.

NHIỆM VỤ: Trích xuất INSIGHTS (hiểu biết sâu sắc) từ cuộc họp

INSIGHTS LÀ:
- Phát hiện quan trọng, xu hướng, vấn đề được nhấn mạnh
- Nguyên nhân gốc rễ của vấn đề
- Cơ hội hoặc rủi ro được xác định
- Mâu thuẫn/gap trong kế hoạch
- Bài học/best practices

KHÔNG PHẢI:
- Chỉ tóm tắt lại nội dung
- Ý kiến không có dữ liệu
- Thông tin quá chung chung

OUTPUT: Bullet points (bắt đầu bằng "- ")
Mỗi insight: 15-30 từ
Số lượng: 3-7 insights"""

        chunks = self._chunk_context(context, chunk_size=6000)
        all_insights = []
        
        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                print(f"         Chunk {i+1}/{len(chunks)}...")
            
            user_prompt = f"""Trích xuất insights:

{chunk}

INSIGHTS:"""
            
            try:
                response = chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    options={
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1
                    }
                )
                
                content = response["message"]["content"]
                
                for line in content.split('\n'):
                    line = line.strip()
                    if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                        insight = line[1:].strip()
                        if len(insight) > 20:
                            all_insights.append(insight)
            
            except Exception as e:
                print(f"         ⚠️  Error in chunk {i+1}: {e}")
                continue
        
        # Deduplicate
        unique = []
        seen = set()
        
        for insight in all_insights:
            key = insight.lower()
            if not any(key in s or s in key for s in seen):
                unique.append(insight)
                seen.add(key)
        
        return unique[:7]
    
    def _sanitize_json(self, json_str: str) -> str:
        """
        Sanitize JSON string to fix common LLM errors
        
        Fixes:
        - Unescaped quotes inside strings
        - Newlines in strings (both literal and escaped)
        - Trailing commas
        - Single quotes instead of double quotes
        - Unicode escapes
        """
        # STEP 1: Fix structural issues
        # Remove trailing commas before ] or }
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        # STEP 2: Fix string content issues - BEFORE collapsing whitespace
        # Replace actual newline characters with escaped version
        # This is crucial - LLM often puts real newlines in strings
        json_str = json_str.replace('\r\n', '\\n')  # Windows newlines
        json_str = json_str.replace('\n', '\\n')     # Unix newlines
        json_str = json_str.replace('\r', '\\n')     # Old Mac newlines
        
        # Now unescape them to spaces (more readable)
        json_str = json_str.replace('\\n', ' ')
        
        # STEP 3: Normalize whitespace
        json_str = re.sub(r'\s+', ' ', json_str)  # Multiple spaces → single space
        json_str = json_str.strip()
        
        # STEP 4: Fix quotes
        # Replace single quotes with double (but not in contractions)
        # This is a simple approach - might need refinement
        # json_str = json_str.replace("'", '"')  # Disabled - too aggressive
        
        return json_str
    
    def _fallback_parse_tasks(self, json_str: str) -> List[Dict]:
        """
        Fallback parser when JSON parsing fails
        
        Strategy:
        1. Try to manually fix the JSON and re-parse
        2. If that fails, use regex to extract fields
        """
        tasks = []
        
        # ATTEMPT 1: More aggressive JSON fixing
        try:
            # Remove all literal newlines and replace with space
            fixed_json = json_str
            
            # Fix common patterns where LLM breaks JSON:
            # Pattern: "source": "text\n  },
            # Should be: "source": "text" },
            
            # Find all string values and fix them
            def fix_string_value(match):
                key = match.group(1)
                value = match.group(2)
                # Remove newlines and extra spaces
                value = value.replace('\n', ' ').replace('\r', ' ')
                value = re.sub(r'\s+', ' ', value).strip()
                return f'"{key}": "{value}"'
            
            # Pattern: "key": "value with possible newlines"
            fixed_json = re.sub(r'"(\w+)"\s*:\s*"([^"]*?)"', fix_string_value, fixed_json, flags=re.DOTALL)
            
            # Try parsing fixed JSON
            tasks = json.loads(fixed_json)
            if isinstance(tasks, list) and len(tasks) > 0:
                print(f"            ✓ Recovered via aggressive JSON repair")
                return self._validate_tasks(tasks)
        except Exception as e:
            print(f"            ⚠️  Aggressive repair failed: {e}")
        
        # ATTEMPT 2: Regex extraction (last resort)
        try:
            # Split into individual task objects more carefully
            # Look for opening { followed by "task"
            task_pattern = r'\{\s*"task"\s*:\s*"([^"]+)"[^}]*?"assigned_to"\s*:\s*(?:"([^"]+)"|null)[^}]*?"deadline"\s*:\s*(?:"([^"]+)"|null)[^}]*?"priority"\s*:\s*"([^"]+)"[^}]*?"how_to"\s*:\s*"([^"]+)"'
            
            matches = re.finditer(task_pattern, json_str, re.DOTALL)
            
            for match in matches:
                task = {
                    'task': match.group(1).strip(),
                    'assigned_to': match.group(2).strip() if match.group(2) else None,
                    'deadline': match.group(3).strip() if match.group(3) else None,
                    'priority': match.group(4).strip() if match.group(4) else 'medium',
                    'how_to': match.group(5).strip() if match.group(5) else 'Chi tiết sẽ được bổ sung.'
                }
                tasks.append(task)
            
            if tasks:
                print(f"            ✓ Recovered {len(tasks)} tasks via regex extraction")
        except Exception as e:
            print(f"            ⚠️  Regex extraction failed: {e}")
        
        # Validate extracted tasks
        if tasks:
            return self._validate_tasks(tasks)
        return []
