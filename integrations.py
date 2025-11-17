"""
Integration với Trello, Notion, ClickUp
"""
import os
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
from trello import TrelloClient
from notion_client import Client as NotionClient


# ======================= TRELLO =======================
class TrelloIntegration:
    def __init__(self, api_key: str, token: str, board_id: str):
        self.client = TrelloClient(api_key=api_key, token=token)
        self.board_id = board_id
        self.board = self.client.get_board(board_id)
    
    def create_card_from_task(self, task: Dict[str, Any], list_name: str = "To Do") -> str:
        """
        Tạo Trello card từ task
        
        Args:
            task: {task, assigned_to, deadline, priority}
            list_name: Tên list trong board (To Do, In Progress, Done)
        
        Returns:
            Card URL
        """
        # Tìm list
        lists = self.board.list_lists()
        target_list = None
        for lst in lists:
            if lst.name.lower() == list_name.lower():
                target_list = lst
                break
        
        if not target_list:
            # Tạo list mới nếu chưa có
            target_list = self.board.add_list(list_name)
        
        # Tạo description
        desc = f"**Assigned to:** {task.get('assigned_to') or 'Chưa rõ'}\n"
        desc += f"**Deadline:** {task.get('deadline') or 'Chưa rõ'}\n"
        desc += f"**Priority:** {task.get('priority', 'medium').upper()}\n\n"
        if task.get('context'):
            desc += f"**Context:** {task['context']}"
        
        # Tạo card
        card = target_list.add_card(
            name=task['task'],
            desc=desc
        )
        
        # Thêm label theo priority
        priority_colors = {
            'high': 'red',
            'medium': 'orange',
            'low': 'green'
        }
        color = priority_colors.get(task.get('priority', 'medium'), 'yellow')
        
        # Trello labels
        labels = self.board.get_labels()
        priority_label = None
        for label in labels:
            if label.color == color:
                priority_label = label
                break
        
        if not priority_label:
            priority_label = self.board.add_label(
                name=task.get('priority', 'medium').upper(),
                color=color
            )
        
        card.add_label(priority_label)
        
        return card.url
    
    def export_tasks(self, tasks: List[Dict], summary: str = None) -> List[str]:
        """
        Export tất cả tasks sang Trello
        
        Returns:
            List of card URLs
        """
        card_urls = []
        
        # Tạo card cho summary nếu có
        if summary:
            lists = self.board.list_lists()
            info_list = None
            for lst in lists:
                if "info" in lst.name.lower() or "summary" in lst.name.lower():
                    info_list = lst
                    break
            
            if not info_list:
                info_list = self.board.add_list("📋 Summary")
            
            summary_card = info_list.add_card(
                name=f"Meeting Summary - {datetime.now().strftime('%Y-%m-%d')}",
                desc=summary
            )
            card_urls.append(summary_card.url)
        
        # Tạo cards cho tasks
        for task in tasks:
            priority = task.get('priority', 'medium')
            list_name = "🔴 High Priority" if priority == 'high' else "To Do"
            
            try:
                url = self.create_card_from_task(task, list_name)
                card_urls.append(url)
            except Exception as e:
                print(f"⚠️  Lỗi tạo card: {e}")
        
        return card_urls


# ======================= NOTION =======================
class NotionIntegration:
    def __init__(self, api_key: str, database_id: str):
        self.client = NotionClient(auth=api_key)
        self.database_id = database_id
    
    def create_page_from_task(self, task: Dict[str, Any]) -> str:
        """
        Tạo Notion page từ task
        """
        properties = {
            "Name": {
                "title": [{"text": {"content": task['task']}}]
            },
            "Status": {
                "select": {"name": "To Do"}
            },
            "Priority": {
                "select": {"name": task.get('priority', 'medium').capitalize()}
            }
        }
        
        # Thêm assigned_to nếu có
        if task.get('assigned_to'):
            properties["Assigned"] = {
                "rich_text": [{"text": {"content": task['assigned_to']}}]
            }
        
        # Thêm deadline nếu có
        if task.get('deadline'):
            properties["Deadline"] = {
                "rich_text": [{"text": {"content": task['deadline']}}]
            }
        
        # Tạo page
        page = self.client.pages.create(
            parent={"database_id": self.database_id},
            properties=properties
        )
        
        # Thêm context vào body nếu có
        if task.get('context'):
            self.client.blocks.children.append(
                block_id=page['id'],
                children=[{
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"text": {"content": task['context']}}]
                    }
                }]
            )
        
        return page['url']
    
    def export_tasks(self, tasks: List[Dict], summary: str = None) -> List[str]:
        """
        Export tasks sang Notion database
        """
        page_urls = []
        
        # Tạo summary page nếu có
        if summary:
            try:
                summary_page = self.client.pages.create(
                    parent={"database_id": self.database_id},
                    properties={
                        "Name": {
                            "title": [{"text": {"content": f"Meeting Summary - {datetime.now().strftime('%Y-%m-%d')}"}}]
                        },
                        "Status": {
                            "select": {"name": "Done"}
                        }
                    }
                )
                
                # Thêm summary content
                self.client.blocks.children.append(
                    block_id=summary_page['id'],
                    children=[{
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"text": {"content": summary}}]
                        }
                    }]
                )
                
                page_urls.append(summary_page['url'])
            except Exception as e:
                print(f"⚠️  Lỗi tạo summary page: {e}")
        
        # Tạo pages cho tasks
        for task in tasks:
            try:
                url = self.create_page_from_task(task)
                page_urls.append(url)
            except Exception as e:
                print(f"⚠️  Lỗi tạo task page: {e}")
        
        return page_urls


# ======================= CLICKUP =======================
class ClickUpIntegration:
    def __init__(self, api_key: str, list_id: str):
        self.api_key = api_key
        self.list_id = list_id
        self.base_url = "https://api.clickup.com/api/v2"
        self.headers = {
            "Authorization": api_key,
            "Content-Type": "application/json"
        }
    
    def create_task_from_task(self, task: Dict[str, Any]) -> str:
        """
        Tạo ClickUp task
        """
        # Map priority
        priority_map = {
            'high': 1,
            'medium': 3,
            'low': 4
        }
        
        payload = {
            "name": task['task'],
            "description": f"**Assigned:** {task.get('assigned_to') or 'Chưa rõ'}\n"
                          f"**Deadline:** {task.get('deadline') or 'Chưa rõ'}\n\n"
                          f"{task.get('context', '')}",
            "priority": priority_map.get(task.get('priority', 'medium'), 3),
            "status": "to do"
        }
        
        # Thêm assignees nếu có (cần user_id thực tế)
        # payload["assignees"] = [user_id]
        
        response = requests.post(
            f"{self.base_url}/list/{self.list_id}/task",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code == 200:
            task_data = response.json()
            return task_data['url']
        else:
            raise Exception(f"ClickUp API error: {response.status_code} - {response.text}")
    
    def export_tasks(self, tasks: List[Dict], summary: str = None) -> List[str]:
        """
        Export tasks sang ClickUp
        """
        task_urls = []
        
        # Tạo task cho summary nếu có
        if summary:
            try:
                summary_payload = {
                    "name": f"📋 Meeting Summary - {datetime.now().strftime('%Y-%m-%d')}",
                    "description": summary,
                    "status": "complete"
                }
                
                response = requests.post(
                    f"{self.base_url}/list/{self.list_id}/task",
                    headers=self.headers,
                    json=summary_payload
                )
                
                if response.status_code == 200:
                    task_urls.append(response.json()['url'])
            except Exception as e:
                print(f"⚠️  Lỗi tạo summary task: {e}")
        
        # Tạo tasks
        for task in tasks:
            try:
                url = self.create_task_from_task(task)
                task_urls.append(url)
            except Exception as e:
                print(f"⚠️  Lỗi tạo task: {e}")
        
        return task_urls


# ======================= HELPER FUNCTIONS =======================
def get_trello_client(api_key: str, token: str, board_id: str) -> Optional[TrelloIntegration]:
    """Khởi tạo Trello client"""
    try:
        return TrelloIntegration(api_key, token, board_id)
    except Exception as e:
        print(f"❌ Lỗi kết nối Trello: {e}")
        return None


def get_notion_client(api_key: str, database_id: str) -> Optional[NotionIntegration]:
    """Khởi tạo Notion client"""
    try:
        return NotionIntegration(api_key, database_id)
    except Exception as e:
        print(f"❌ Lỗi kết nối Notion: {e}")
        return None


def get_clickup_client(api_key: str, list_id: str) -> Optional[ClickUpIntegration]:
    """Khởi tạo ClickUp client"""
    try:
        return ClickUpIntegration(api_key, list_id)
    except Exception as e:
        print(f"❌ Lỗi kết nối ClickUp: {e}")
        return None
