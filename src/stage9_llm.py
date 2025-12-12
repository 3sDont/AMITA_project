"""
Stage 9: LLM Analysis
Generate summary, extract tasks, insights
"""
from typing import Dict, List
import json
import re
try:
    from ollama import chat, list as ollama_list
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class LLMAnalyzer:
    """
    LLM Analyzer - Phân tích meeting bằng LLM
    
    Features:
        - Summary generation
        - Task extraction với how_to
        - Key insights
    """
    
    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if Ollama and model are available"""
        if not OLLAMA_AVAILABLE:
            return False
        
        try:
            # Try to list models
            models = ollama_list()
            model_names = [m['name'] for m in models.get('models', [])]
            # Check if our model exists (with or without tag)
            model_exists = any(
                self.model in name or self.model.split(':')[0] in name 
                for name in model_names
            )
            return model_exists
        except Exception:
            return False
        self.available = self._check_availability()
    
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
        # Check availability
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
    
    def _build_context(self, segments: List[Dict], speakers: Dict, metadata: Dict) -> str:
        """Build context string from segments"""
        lines = []
        
        # Add metadata
        lines.append(f"Meeting: {metadata.get('meeting_id', 'Unknown')}")
        lines.append(f"Duration: {len(segments)} segments\n")
        
        # Add transcript
        for seg in segments:
            speaker = seg.get('speaker_display', seg['speaker'])
            gender_emoji = "👨" if speakers.get(seg['speaker'], {}).get('gender') == 'Male' else "👩"
            lines.append(f"{gender_emoji} {speaker}: {seg['text']}")
        
        return "\n".join(lines)
    
    def _generate_summary(self, context: str) -> str:
        """Generate meeting summary"""
        prompt = f"""Tóm tắt cuộc họp sau (100-150 từ):

{context[:3000]}

Tóm tắt:"""
        
        response = chat(model=self.model, messages=[
            {"role": "user", "content": prompt}
        ])
        
        return response["message"]["content"].strip()
    
    def _extract_tasks(self, context: str) -> List[Dict]:
        """Extract tasks với how_to"""
        prompt = f"""Trích xuất tasks từ cuộc họp. Format JSON:

[{{"task": "...", "assigned_to": "...", "deadline": "...", "how_to": "..."}}]

Context:
{context[:3000]}

JSON:"""
        
        response = chat(model=self.model, messages=[
            {"role": "user", "content": prompt}
        ])
        
        # Parse JSON
        try:
            content = response["message"]["content"]
            # Extract JSON array
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                tasks = json.loads(match.group())
                # Validate
                for task in tasks:
                    if 'task' not in task:
                        continue
                    if 'how_to' not in task or not task['how_to']:
                        task['how_to'] = "Thực hiện theo quy trình chuẩn."
                    if 'priority' not in task:
                        task['priority'] = 'medium'
                return tasks
        except:
            pass
        
        return []
    
    def _extract_insights(self, context: str) -> List[str]:
        """Extract key insights"""
        prompt = f"""Liệt kê 3-5 insights quan trọng từ cuộc họp:

{context[:2000]}

Insights:"""
        
        response = chat(model=self.model, messages=[
            {"role": "user", "content": prompt}
        ])
        
        content = response["message"]["content"]
        
        # Parse insights (assuming bullet points)
        insights = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                insights.append(line[1:].strip())
        
        return insights[:5]
