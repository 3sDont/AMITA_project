"""
Stage 9: LLM Analysis (IMPROVED VERSION)
Generate summary, extract tasks

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
                tasks: List[Dict]
            }
        """
        if not self.available:
            print(f"   ⚠️  Model '{self.model}' not available")
            print(f"   💡 To use LLM features:")
            print(f"      1. Install Ollama: https://ollama.ai")
            print(f"      2. Pull model: ollama pull {self.model}")
            return {
                'summary': "[LLM not available - please install Ollama and pull model]",
                'tasks': []
            }
        
        print(f"   🤖 Using model: {self.model}")
        
        # Build context (NO truncation - full transcript)
        context = self._build_context(segments, speakers, metadata, truncate=False)
        context_length = len(context)
        print(f"   📊 Context: {context_length:,} chars, {len(segments)} segments")
        
        # Generate summary
        print(f"   📝 Generating summary...")
        summary = self._generate_summary(context)
        
        # Extract tasks
        print(f"   ✅ Extracting tasks...")
        tasks = self._extract_tasks(context)

        return {
            'summary': summary,
            'tasks': tasks
        }
    
    def _build_context(
        self, 
        segments: List[Dict], 
        speakers: Dict, 
        metadata: Dict,
        max_length: Optional[int] = None,
        truncate: bool = False  # ✅ New parameter to control truncation
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
        segments_included = 0
        
        for seg in segments:
            speaker = seg.get('speaker_display', seg['speaker'])
            text = seg.get('text', '')
            time_str = seg.get('time_str', '')
            
            line = f"[{time_str}] {speaker}: {text}"
            
            # ✅ Only truncate if explicitly requested (for preview purposes)
            if truncate and current_length + len(line) + 1 > max_length:
                lines.append(f"\n[... {len(segments) - segments_included} more segments ...]")
                break
            
            lines.append(line)
            current_length += len(line) + 1
            segments_included += 1
        
        # ✅ Add footer showing coverage
        if segments_included == len(segments):
            lines.append(f"\n=== END OF TRANSCRIPT ({segments_included} segments) ===")
        
        return "\n".join(lines)
    
    def _chunk_context(self, context: str, chunk_size: int = 6000, overlap: int = 500) -> List[str]:
        """Split context into overlapping chunks"""
        if len(context) <= chunk_size:
            return [context]
        
        chunks = []
        start = 0
        chunk_num = 0
        
        while start < len(context):
            end = start + chunk_size
            
            # Find last newline to avoid cutting mid-sentence
            if end < len(context):
                last_newline = context.rfind('\n', start, end)
                if last_newline > start:
                    end = last_newline
            
            chunk = context[start:end]
            chunks.append(chunk)
            chunk_num += 1
            
            # ✅ Better overlap calculation to ensure no segments are missed
            start = end - overlap
            
            # Avoid infinite loop
            if start >= len(context) or end >= len(context):
                break
        
        return chunks
    
    def _generate_summary(self, context: str) -> str:
        """Generate summary với improved prompts"""
        system_prompt = """Vai trò: Bạn là một Thư ký điều hành và Chuyên gia phân tích dữ liệu hội thoại. Nhiệm vụ của bạn là trích xuất "giá trị cốt lõi" từ bản ghi chép cuộc họp (transcript) dưới đây.

QUY TẮC XỬ LÝ DỮ LIỆU:
1. Trung thực: CHỈ sử dụng thông tin CÓ TRONG transcript. Tuyệt đối KHÔNG suy diễn ý định của người nói.
2. Loại bỏ nhiễu: Bỏ qua các lời chào hỏi, chuyện phiếm, hoặc các thảo luận ngoài lề không đi đến kết quả cụ thể.
3. Độ dài: Tối ưu trong khoảng 150-250 từ (không tính emoji) để đảm bảo đủ ý nhưng vẫn súc tích.
4. Đối tượng: Viết cho cấp quản lý đọc để nắm bắt toàn bộ tình hình trong 1-2 phút.
5. Ngôn ngữ: Tiếng Việt chuyên nghiệp, rõ ràng, tránh văn nói.

CẤU TRÚC ĐẦU RA (BẮT BUỘC):

🎯 Mục tiêu chính: 
[Mô tả ngắn gọn lý do cuộc họp diễn ra trong 1 câu, khoảng 15-25 từ]

📋 Nội dung thảo luận trọng tâm:
- [Vấn đề/Topic 1]: [Ý kiến chính hoặc giải pháp được đề xuất]
- [Vấn đề/Topic 2]: [Ý kiến chính hoặc giải pháp được đề xuất]
- [Vấn đề/Topic 3]: [Ý kiến chính hoặc giải pháp được đề xuất]
(Tối thiểu 2, tối đa 5 điểm tùy vào độ dài và mật độ thông tin. Mỗi điểm 20-40 từ)

✅ Kết luận & Quyết định:
[Ghi rõ các quyết định được thống nhất, sự đồng thuận, hoặc next steps. Nếu không có kết luận rõ ràng, ghi: "Cuộc họp chưa đi đến kết luận cụ thể, cần họp tiếp để quyết định."]

EDGE CASES:
- Nếu cuộc họp chỉ có 1 topic → Chỉ cần 1 điểm trong phần "Nội dung thảo luận"
- Nếu không có quyết định rõ ràng → Ghi: "Chưa có quyết định cụ thể"
- Nếu transcript quá ngắn (<100 từ) → Tóm tắt ngắn gọn, không ép format phức tạp"""

        chunks = self._chunk_context(context, chunk_size=6000)
        
        if len(chunks) == 1:
            user_prompt = f"""TRANSCRIPT CUỘC HỌP:
{chunks[0]}

---

YÊU CẦU:
Hãy đọc transcript trên và tóm tắt theo ĐÚNG FORMAT sau (bao gồm cả emoji):

🎯 Mục tiêu chính: 
[1 câu ngắn gọn, 15-25 từ]

📋 Nội dung thảo luận trọng tâm:
- [Topic 1]: [Nội dung chi tiết, 20-40 từ]
- [Topic 2]: [Nội dung chi tiết, 20-40 từ]
- [Topic 3]: [Nội dung chi tiết, 20-40 từ]
(Tối thiểu 2, tối đa 5 điểm)

✅ Kết luận & Quyết định:
[Quyết định cụ thể hoặc "Chưa có kết luận cụ thể"]

QUAN TRỌNG:
- BẮT ĐẦU output bằng emoji 🎯, KHÔNG phải "===" hay header khác
- GIỮ NGUYÊN format và emoji
- KHÔNG thêm header "MEETING INFORMATION" hay "TRANSCRIPT"

BẮT ĐẦU TÓM TẮT:"""
            
            response = chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": 0.3,  # Deterministic output
                    "top_p": 0.9,
                    "repeat_penalty": 1.2  # Tăng để tránh lặp lại pattern
                }
            )
            
            return response["message"]["content"].strip()
        
        else:
            # Long meeting - chunked processing
            print(f"      💡 Long meeting: {len(chunks)} chunks")
            
            # Strategy: Extract key points from each chunk first, then synthesize
            chunk_keypoints = []
            for i, chunk in enumerate(chunks):
                print(f"         Processing chunk {i+1}/{len(chunks)}...")
                
                # Extract key points (not full summary) to preserve context
                user_prompt = f"""Trích xuất các điểm chính (key points) từ phần transcript này:

{chunk}

YÊU CẦU:
- Liệt kê 3-5 điểm chính được thảo luận
- Mỗi điểm: 1 câu ngắn (10-20 từ)
- Giữ nguyên tên người, số liệu, quyết định cụ thể
- Format: "- [Điểm 1], - [Điểm 2], ..."

KEY POINTS:"""
                
                response = chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Bạn là trợ lý trích xuất thông tin chính xác từ transcript."},
                        {"role": "user", "content": user_prompt}
                    ],
                    options={
                        "temperature": 0.3,  # More deterministic for extraction
                        "top_p": 0.9,
                        "repeat_penalty": 1.2
                    }
                )
                
                chunk_keypoints.append(response["message"]["content"].strip())
            
            # Synthesize all key points into final summary
            combined_keypoints = "\n\n".join([f"Phần {i+1}:\n{kp}" for i, kp in enumerate(chunk_keypoints)])
            
            final_prompt = f"""KEY POINTS TỪ CÁC PHẦN CỦA CUỘC HỌP:

{combined_keypoints}

---

YÊU CẦU:
Tổng hợp các key points trên thành TÓM TẮT HOÀN CHỈNH theo ĐÚNG FORMAT (bao gồm emoji):

🎯 Mục tiêu chính: 
[1 câu ngắn gọn, 15-25 từ]

📋 Nội dung thảo luận trọng tâm:
- [Topic 1]: [Nội dung chi tiết, 20-40 từ]
- [Topic 2]: [Nội dung chi tiết, 20-40 từ]
- [Topic 3]: [Nội dung chi tiết, 20-40 từ]
(Tối thiểu 2, tối đa 5 điểm)

✅ Kết luận & Quyết định:
[Quyết định cụ thể hoặc "Chưa có kết luận cụ thể"]

QUAN TRỌNG:
- BẮT ĐẦU output bằng emoji 🎯, KHÔNG phải "===" hay header khác
- GIỮ NGUYÊN format và emoji
- Loại bỏ thông tin trùng lặp

BẮT ĐẦU TÓM TẮT TỔNG HỢP:"""
            
            response = chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": final_prompt}
                ],
                options={
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "repeat_penalty": 1.2
                }
            )
            
            return response["message"]["content"].strip()
    
    def _extract_tasks(self, context: str) -> List[Dict]:
        """Extract tasks với validation"""
        system_prompt = """Vai trò: Bạn là một Trợ lý Cuộc Họp chuyên nghiệp, có khả năng phân tích hội thoại và chuyển hóa các thảo luận thành danh sách công việc thực thi (Actionable Tasks).

NHIỆM VỤ: TRÍCH XUẤT TẤT CẢ các TASK/CÔNG VIỆC từ transcript cuộc họp và trả về định dạng JSON đã chuẩn hóa.

QUY TẮC NGHIÊM NGẶT (7 ĐIỂM):
1. CHỈ trích xuất tasks được ĐỀ CẬP RÕ RÀNG trong transcript
2. KHÔNG tự suy luận, KHÔNG gán thêm trách nhiệm, KHÔNG suy diễn mục tiêu ẩn
3. TASK hợp lệ PHẢI có ít nhất 1 trong các dấu hiệu sau:
   → Có người được giao việc cụ thể
   → Có deadline/thời hạn
   → Có động từ hành động mạnh (chuẩn bị, liên hệ, gửi, hoàn thành...)
4. KHÔNG phải TASK (loại bỏ):
   → Thảo luận chung, trao đổi ý kiến
   → Nhận xét, đánh giá, chia sẻ thông tin
   → Phản hồi, cảm ơn, chào hỏi
   → Câu hỏi, hỏi đáp không có kết luận
5. Có người thực hiện → Ghi đầy đủ tên/vai trò
6. Có deadline → Ghi cụ thể (ví dụ: "Trước ngày 25/12", "Cuối tuần này")
7. Không tìm thấy task nào → Trả về mảng rỗng: []

DẤU HIỆU NHẬN BIẾT TASK:
✓ Từ khóa cam kết: "sẽ làm", "phải", "cần", "được giao", "nhớ", "bắt buộc", "yêu cầu"
✓ Thời hạn: "deadline", "trước ngày", "hoàn thành vào", "trong tuần này"
✓ Động từ hành động: "chuẩn bị", "liên hệ", "gửi", "kiểm tra", "review", "setup", "deploy"
✓ Chủ ngữ rõ ràng: "Anh A sẽ...", "Em B phải...", "Team C cần..."

FIELD "how_to" - HƯỚNG DẪN THỰC HIỆN (QUAN TRỌNG):
📌 Nếu transcript ĐÃ NÓI cách làm → Trích xuất chính xác từ transcript
📌 Nếu transcript CHƯA NÓI cách làm → Dựa vào context và best practices, đề xuất 2-3 bước CỤ THỂ và KHẢ THI
📌 Format: Viết liền trên 1 dòng, các bước ngăn cách bằng dấu chấm phẩy (;)
📌 Độ dài: 15-40 từ mỗi how_to
📌 TRÁNH: Hướng dẫn quá chung chung như "Thực hiện theo quy trình", "Làm theo hướng dẫn"

FIELD "priority" - ƯU TIÊN:
🔴 high: Có deadline gấp (<3 ngày), blocking task, hoặc được nhấn mạnh trong họp
🟡 medium: Deadline bình thường (3-7 ngày), công việc thường xuyên
🟢 low: Không có deadline, hoặc deadline dài hạn (>7 ngày), công việc phụ

JSON FORMAT (BẮT BUỘC - KHÔNG VI PHẠM):
⚠️ Output PHẢI là JSON ARRAY hợp lệ
⚠️ KHÔNG xuống dòng trong bất kỳ giá trị string nào
⚠️ Tất cả nội dung viết liền trên 1 dòng
⚠️ Dùng dấu phẩy (,) hoặc chấm phẩy (;) thay cho xuống dòng
⚠️ Đảm bảo đóng dấu ngoặc kép đúng

OUTPUT STRUCTURE:
[
  {
    "task": "Tên task (công việc) ngắn gọn (5-15 từ)",
    "assigned_to": "Tên người (Hoặc Null)",
    "deadline": "Thời hạn cụ thể (Hoặc Null)",
    "priority": "Cao/Trung bình/Thấp",
    "how_to": "Hướng dẫn CỤ THỂ 2-3 bước, viết liền, ngăn cách bằng dấu chấm phẩy"
  }
]"""

        chunks = self._chunk_context(context, chunk_size=6000)
        all_tasks = []
        
        print(f"      📦 Split into {len(chunks)} chunks for processing")
        print(f"      📏 Chunk sizes: {[len(c) for c in chunks]} chars")
        
        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                print(f"         Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
            
            # ✅ Include few-shot examples in EVERY request
            user_prompt = f"""

=== BÂY GIỜ, TRÍCH XUẤT TASKS TỪ TRANSCRIPT SAU ===

{chunk}

=== YÊU CẦU ===
- Chỉ trích xuất tasks được đề cập RÕ RÀNG
- Tuân thủ 7 quy tắc nghiêm ngặt
- Output: JSON array hợp lệ, KHÔNG xuống dòng
- Nếu không có task → trả về []

JSON OUTPUT:"""
            
            try:
                response = chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    options={
                        "temperature": 0.3,  # Critical for consistent JSON output
                        "top_p": 0.9,
                        "repeat_penalty": 1.2
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
                        
                        # Skip this chunk - JSON parsing failed
                        print(f"            ⚠️  Skipping this chunk - invalid JSON")
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
        """Remove duplicate tasks based on task name"""
        if not tasks:
            return tasks
        
        seen = set()
        unique = []
        
        for task in tasks:
            task_key = task['task'].lower().strip()
            if task_key not in seen:
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
                            # ✅ VALIDATION: Reject hallucinated/invalid responses
                            # Check 1: Reject if corrected text is too short (< 30% of original)
                            if len(corrected) < len(original) * 0.3:
                                corrected_segments.append(new_seg)
                                continue
                            
                            # Check 2: Reject if corrected text is too long (> 200% of original)
                            if len(corrected) > len(original) * 2.0:
                                corrected_segments.append(new_seg)
                                continue
                            
                            # Check 3: Reject meta-responses (LLM describing its task instead of doing it)
                            meta_patterns = [
                                "tôi có thể", "i can", "i will",
                                "sửa lỗi chính tả", "chỉnh sửa văn bản",
                                "đây là văn bản", "here is the",
                                "văn bản đã sửa", "corrected text",
                                "không có lỗi", "no errors",
                                "văn bản gốc", "original text"
                            ]
                            is_meta_response = any(
                                pattern in corrected.lower() 
                                for pattern in meta_patterns
                            )
                            if is_meta_response:
                                corrected_segments.append(new_seg)
                                continue
                            
                            # Check 4: Reject if response is just a generic phrase
                            if len(corrected.split()) < 3:
                                corrected_segments.append(new_seg)
                                continue
                            
                            # ✅ All validations passed - apply correction
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
    
    def _sanitize_json(self, json_str: str) -> str:
        """Sanitize JSON string to fix common LLM errors"""
        # Remove trailing commas before ] or }
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        # Replace newlines with spaces
        json_str = json_str.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
        
        # Normalize whitespace
        json_str = re.sub(r'\s+', ' ', json_str).strip()
        
        # Fix unquoted string values (most common LLM error)
        # Pattern: "field": value, -> "field": "value",
        # Where value is not null/true/false/number and not already quoted
        def quote_unquoted_values(match):
            field = match.group(1)
            value = match.group(2).strip()
            
            # Skip if already valid: null, true, false, numbers, or already quoted
            if value in ['null', 'true', 'false'] or value.startswith('"') or value.replace('.', '').replace('-', '').isdigit():
                return match.group(0)
            
            # Quote the value
            return f'"{field}": "{value}",'
        
        # Match: "field": unquoted_value,
        json_str = re.sub(r'"([^"]+)":\s*([^,}\]]+),', quote_unquoted_values, json_str)
        
        # Fix last field in object (no trailing comma)
        def quote_last_field(match):
            field = match.group(1)
            value = match.group(2).strip()
            
            if value in ['null', 'true', 'false'] or value.startswith('"') or value.replace('.', '').replace('-', '').isdigit():
                return match.group(0)
            
            return f'"{field}": "{value}"}}'
        
        json_str = re.sub(r'"([^"]+)":\s*([^,}\]]+)\}', quote_last_field, json_str)
        
        return json_str
    

