from utils import *
import chromadb
import uuid
import time
import os
import random
from typing import List
from dataclasses import asdict

TEXT_EMB_DIM = config.get('text_embed_dim')

@dataclass
class EventMemory:
    id: str
    text: str
    type: str
    weight: int
    created_at: float
    distance: float = 0.0
    adjusted_distance: float = 0.0
    time_penalty: float = 0.0

@dataclass
class UserMemory:
    text: str

@dataclass
class SelfMemory:
    id: str
    text: str
    time: float


class MemorySystem:
    def __init__(self, data_dir: str, group_id: int):
        os.makedirs(data_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=os.path.join(data_dir, f"memory_chromadb_{group_id}"))
        self.em_collection = self.client.get_or_create_collection(name="event_memroy")
        self.file_db = get_file_db(os.path.join(data_dir, f"memory_{group_id}.json")) 

    def em_add(self, text: str, embedding: List[float], initial_weight: int) -> str:
        """
        添加一条新的事件记忆，默认为短期记忆。  

        参数:
            text (str): 文本。
            embedding (List[float]): 文本对应的embedding向量。
            initial_weight (int): 记忆的初始权重。
        返回:
            str: 该条记忆的唯一ID。
        """
        assert len(embedding) == TEXT_EMB_DIM, f"Embedding维度应为 {TEXT_EMB_DIM}，但收到 {len(embedding)}"
        memory_id = str(uuid.uuid4())
        self.em_collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            metadatas=[{
                "text": text,
                "type": "short_term",
                "weight": initial_weight,
                "created_at": time.time(),
            }]
        )
        info(f"成功添加事件记忆 {memory_id} 内容: \"{text}\"")
        return memory_id

    def em_increase_weight(self, memory_id: str, weight_increase: int, threshold: int):
        """
        为指定ID的事件记忆增加权重。当权重达到阈值时，转换为长期记忆。

        参数:
            memory_id (str): 要增加权重的记忆ID。
            weight_increase (int): 每次增加的权重量。
            threshold (int): 转换为长期记忆的权重阈值。
        """
        # 获取当前记忆的元数据
        memory = self.em_collection.get(ids=[memory_id], include=["metadatas"])
        if not memory['ids']:
            raise Exception(f"未找到ID为 {memory_id} 的事件记忆")

        current_meta = memory['metadatas'][0]
        current_weight = current_meta.get('weight', 0)
        
        # 更新权重
        new_weight = current_weight + weight_increase
        current_meta['weight'] = new_weight
        
        info(f"事件记忆 {memory_id} 权重 {current_weight} -> {new_weight}")

        # 检查是否达到阈值
        if new_weight >= threshold and current_meta.get('type') == 'short_term':
            current_meta['type'] = 'long_term'
            info(f"记忆 {memory_id} 已转换为长期记忆。")

        # 将更新后的元数据写回数据库
        self.em_collection.update(ids=[memory_id], metadatas=[current_meta])

    def em_query(
        self, 
        query_embedding: List[float], 
        n_results: int, 
        memory_type: str,
        time_decay_rate: float = 0.0,
    ) -> List[EventMemory]:
        """
        查询最相关的记忆。

        参数:
            query_embedding (List[float]): 用于查询的embedding向量。
            n_results (int): 希望返回的结果数量。
            memory_type (str): 查询类型, 可选 "all", "short_term", "long_term"。
            time_decay_rate (float): 短期记忆时间衰减率（每小时增加的距离惩罚）。

        返回:
            List[EMQueryResult]: 相关记忆的查询结果列表
        """
        if n_results <= 0:
            return []
        
        assert len(query_embedding) == TEXT_EMB_DIM, f"Embedding维度应为 {TEXT_EMB_DIM}，但收到 {len(query_embedding)}"

        where_clause = {}
        if memory_type in ["short_term", "long_term"]:
            where_clause = {"type": memory_type}

        # 对于短期记忆，我们需要先获取更多结果，然后手动计算时间衰减
        query_n = n_results * 3 if memory_type == 'short_term' else n_results

        results = self.em_collection.query(
            query_embeddings=[query_embedding],
            n_results=query_n,
            where=where_clause,
            include=["metadatas", "distances"]
        )

        processed_results = []
        
        # ChromaDB返回的结果是嵌套列表，我们需要解包
        ids = results['ids'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]

        for i in range(len(ids)):
            res = EventMemory(
                id=ids[i],
                text=metadatas[i].get('text', ''),
                distance=distances[i],
                adjusted_distance=distances[i],
                type=metadatas[i].get('type', 'unknown'),
                weight=metadatas[i].get('weight', 0),
                created_at=metadatas[i].get('created_at', 0),
                time_penalty=0.0,
            )
            # 如果是短期记忆，计算时间衰减
            if res.type == 'short_term':
                time_elapsed_seconds = time.time() - res.created_at
                time_elapsed_hours = time_elapsed_seconds / 3600
                # 距离越小越相关，所以我们给旧的记忆增加一个距离惩罚
                time_penalty = time_elapsed_hours * time_decay_rate
                res.adjusted_distance = res.distance + time_penalty
                res.time_penalty = time_penalty
            processed_results.append(res)

        # 根据调整后的距离进行排序
        processed_results.sort(key=lambda x: x.adjusted_distance)
        return processed_results[:n_results]

    def em_forget(self, forget_time: float, forget_prob: float):
        """
        遗忘指定时间之前的短期记忆，按概率遗忘。

        参数:
            forget_time (float): 遗忘时间点（时间戳），早于此时间的短期记忆将被考虑遗忘。
            forget_prob (float): 遗忘概率（0到1之间）。
        """
        candidate_memories = self.em_collection.get(
            where={
                "$and": [
                    {"type": "short_term"},
                    {"created_at": {"$lt": forget_time}}
                ]
            }
        )
        if not candidate_memories['ids']:
            return
        # 遍历候选记忆，根据概率决定哪些需要被遗忘
        ids_to_forget = []
        for memory_id in candidate_memories['ids']:
            if random.random() < forget_prob:
                ids_to_forget.append(memory_id)
        # 如果有需要遗忘的记忆，则执行删除操作
        if ids_to_forget:
            info(f"将遗忘 {len(ids_to_forget)} 条短期记忆")
            self.em_collection.delete(ids=ids_to_forget)
            info(f"成功遗忘短期记忆: {ids_to_forget}")
        else:
            info("没有短期记忆被遗忘")

    def um_update(self, user_id: int, um: UserMemory):
        """
        更新用户记忆。

        参数:
            user_id (int): 用户ID。
            um (UserMemory): 记忆对象。
        """
        ums = self.file_db.get('ums', {})
        ums[str(user_id)] = asdict(um)
        self.file_db.set('ums', ums)
        info(f"更新用户记忆 {user_id}: {um}")

    def um_get(self, user_id: int) -> UserMemory | None:
        """
        获取用户记忆。

        参数:
            user_id (int): 用户ID。

        返回:
            UserMemory: 记忆对象。
        """
        ums = self.file_db.get('ums', {})
        if str(user_id) in ums:
            return UserMemory(**ums[str(user_id)])
        return None

    def sm_add(self, msg_id: int, text: str, keep_count: int):
        """
        更新自身对话记忆。

        参数:
            msgs (list[dict]): 新加入的对话消息列表。
            keep_count (int): 保留的消息数量。
        """
        sms = self.file_db.get('sms', [])
        sms.append({
            'id': str(msg_id),
            'text': text,
            'time': datetime.now().timestamp(),
        })
        sms = sms[-keep_count:]
        self.file_db.set('sms', sms)
        info(f"更新自身对话记忆，保留最近 {keep_count} 条消息")

    def sm_get(self) -> list[SelfMemory]:
        """
        获取自身对话记忆。

        返回:
            list[dict]: 对话消息列表。
        """
        return [SelfMemory(**sm) for sm in self.file_db.get('sms', [])]


