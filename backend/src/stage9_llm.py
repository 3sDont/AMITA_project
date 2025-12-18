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
        system_prompt = """Bạn là chuyên gia phân tích cuộc họp chuyên nghiệp.

NHIỆM VỤ: Tóm tắt cuộc họp chính xác và súc tích

QUY TẮC BẮT BUỘC:
- CHỈ tóm tắt nội dung THỰC SỰ được thảo luận trong transcript
- KHÔNG tự suy luận hoặc thêm thông tin không có
- Độ dài: 100-200 từ
- Tập trung: mục đích, nội dung chính, kết luận/quyết định
- Ngôn ngữ: Tiếng Việt, súc tích, chuyên nghiệp

FORMAT OUTPUT:
🎯 Mục đích: [Lý do tổ chức cuộc họp]
📋 Nội dung: [2-3 điểm chính được thảo luận]
✅ Kết luận: [Quyết định/kết luận nếu có]"""

        chunks = self._chunk_context(context, chunk_size=6000)
        
        if len(chunks) == 1:
            user_prompt = f"""Dựa vào transcript cuộc họp, hãy tóm tắt theo format đã cho:

{chunks[0]}

TÓM TẮT:"""
            
            response = chat(model=self.model, messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
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
                
                response = chat(model=self.model, messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ])
                
                chunk_summaries.append(response["message"]["content"].strip())
            
            # Combine
            combined = "\n\n".join([f"Phần {i+1}: {s}" for i, s in enumerate(chunk_summaries)])
            
            final_prompt = f"""Kết hợp các phần tóm tắt thành TÓM TẮT TỔNG HỢP:

{combined}

TÓM TẮT TỔNG HỢP:"""
            
            response = chat(model=self.model, messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_prompt}
            ])
            
            return response["message"]["content"].strip()
    
    def _extract_tasks(self, context: str) -> List[Dict]:
        """Extract tasks với validation"""
        system_prompt = """Bạn là chuyên gia phân tích cuộc họp.

NHIỆM VỤ: TRÍCH XUẤT các TASK/CÔNG VIỆC được đề cập trong cuộc họp

QUY TẮC NGHIÊM NGẶT:
1. CHỈ trích xuất tasks được ĐỀ CẬP RÕ RÀNG trong transcript
2. KHÔNG tự suy luận/thêm tasks
3. Task = HÀNH ĐỘNG CỤ THỂ (không phải thảo luận chung)
4. Có người thực hiện → ghi tên
5. Có deadline → ghi cụ thể
6. Không có task nào → trả về []

DẤU HIỆU TASK:
- Từ khóa: "sẽ làm", "phải làm", "cần làm", "được giao"
- Động từ: "chuẩn bị", "liên hệ", "gửi", "kiểm tra", "hoàn thành"
- Có deadline hoặc người thực hiện

QUAN TRỌNG - FIELD "how_to":
- Nếu cuộc họp ĐÃ NÓI cách thực hiện → Ghi lại chính xác
- Nếu cuộc họp CHƯA NÓI → Dựa vào context và kinh nghiệm, đề xuất 1-2 bước cụ thể
- LUÔN PHẢI có nội dung hữu ích, KHÔNG để trống hoặc chung chung
- Format: "..., ..." hoặc mô tả ngắn gọn trong 1 đến 2 câu

OUTPUT FORMAT (JSON ARRAY):
CHÚ Ý: 
- Mỗi string PHẢI KHÔNG chứa dấu xuống dòng (newline)
- Dùng dấu phẩy hoặc dấu chấm thay vì xuống dòng
- Đảm bảo tất cả dấu ngoặc kép đều được đóng đúng

[
  {
    "task": "Công việc ngắn gọn (5-15 từ, không xuống dòng)",
    "assigned_to": "Tên người hoặc null",
    "deadline": "Thời hạn hoặc null",
    "priority": "high/medium/low",
    "how_to": "Hướng dẫn CỤ THỂ 1-2 câu KHÔNG xuống dòng",
    "source": "Trích dẫn từ transcript"
  }
]"""

        few_shot = """
VÍ DỤ:

Input: "Anh Minh sẽ chuẩn bị báo cáo Q4 trước ngày 15. Em Lan liên hệ team IT setup server."

Output:
[
  {
    "task": "Chuẩn bị báo cáo Q4",
    "assigned_to": "Anh Minh",
    "deadline": "Trước ngày 15",
    "priority": "high",
    "how_to": "Thu thập số liệu doanh thu, chi phí, lợi nhuận Q4. Tạo file báo cáo theo template công ty và gửi approval.",
    "source": "Anh Minh sẽ chuẩn bị báo cáo Q4 trước ngày 15"
  },
  {
    "task": "Liên hệ team IT setup server",
    "assigned_to": "Em Lan",
    "deadline": null,
    "priority": "medium",
    "how_to": "Soạn email mô tả yêu cầu cấu hình server (RAM, CPU, storage), gửi đến it-support@company.com hoặc liên hệ trực tiếp qua Slack.",
    "source": "Em Lan liên hệ team IT setup server"
  }
]

CHÚ Ý: Field "how_to" phải CỤ THỂ và KHẢ THI, giúp người đọc hiểu NGAY cách thực hiện!
"""

        chunks = self._chunk_context(context, chunk_size=6000)
        all_tasks = []
        
        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                print(f"         Chunk {i+1}/{len(chunks)}...")
            
            user_prompt = f"""{few_shot}

Trích xuất tasks từ transcript:

{chunk}

JSON OUTPUT:"""
            
            try:
                response = chat(model=self.model, messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ])
                
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
                        print(f"            🔍 Error location in JSON:")
                        # Show context around error
                        error_pos = je.pos if hasattr(je, 'pos') else 364
                        start = max(0, error_pos - 100)
                        end = min(len(json_str), error_pos + 100)
                        print(f"            {json_str[start:end]}")
                        print(f"            {' ' * (error_pos - start)}^ ERROR HERE")
                        
                        # Try to fix common issues
                        json_str = json_str.replace("'", '"')  # Single quotes to double
                        json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas
                        try:
                            tasks = json.loads(json_str)
                            if isinstance(tasks, list):
                                validated = self._validate_tasks(tasks)
                                all_tasks.extend(validated)
                        except:
                            pass  # Give up on this chunk
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
                'how_to': task.get('how_to', 'Chi tiết thực hiện sẽ được bổ sung.'),
                'source': task.get('source', '')
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
                response = chat(model=self.model, messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ])
                
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
    
