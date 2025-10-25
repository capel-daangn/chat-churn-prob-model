"""
Synthetic Chat Data Generator
Generates realistic buyer-seller conversation data with churn labels
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import yaml


class ChatDataGenerator:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.num_conversations = self.config['data']['num_conversations']
        self.avg_messages = self.config['data']['avg_messages_per_conv']
        self.churn_ratio = self.config['data']['churn_ratio']

        # Message templates for buyer and seller
        self.buyer_messages = {
            'interested': [
                "이 상품 아직 구매 가능한가요?",
                "가격 네고 가능할까요?",
                "실물 사진 더 보여주실 수 있나요?",
                "언제 거래 가능하신가요?",
                "직거래 가능한가요?",
                "상품 상태는 어떤가요?",
                "배송비 포함인가요?",
            ],
            'positive': [
                "네 좋습니다!",
                "알겠습니다. 구매할게요",
                "감사합니다",
                "빠른 답변 감사드려요",
                "그럼 그렇게 진행해주세요",
            ],
            'churn': [
                "음... 좀 비싸네요",
                "죄송합니다만 다시 생각해볼게요",
                "...",
                "고민 좀 해볼게요",
                "다른 상품이랑 비교 좀 해보고 연락드릴게요",
                "아 그렇군요",
            ]
        }

        self.seller_messages = {
            'positive': [
                "네, 구매 가능합니다!",
                "네고 조금 가능합니다",
                "사진 보내드릴게요",
                "오늘 오후 가능합니다",
                "직거래 가능합니다",
                "상품 상태 아주 좋습니다",
                "배송비 별도입니다",
            ],
            'negative': [
                "죄송하지만 네고는 어렵습니다",
                "가격은 정찰제입니다",
                "현재 다른 분과 거래 진행 중입니다",
                "직거래는 어렵습니다",
            ],
            'neutral': [
                "네",
                "확인했습니다",
                "알겠습니다",
            ]
        }

    def generate_conversation(self, conv_id):
        """Generate a single conversation"""
        messages = []
        num_messages = np.random.poisson(self.avg_messages)
        num_messages = max(3, min(num_messages, 20))  # 3~20 messages

        # Decide if this conversation will have churn
        will_churn = random.random() < self.churn_ratio

        base_time = datetime.now() - timedelta(days=random.randint(1, 30))

        for msg_idx in range(num_messages):
            # Alternate between buyer and seller
            sender_role = "buyer" if msg_idx % 2 == 0 else "seller"

            # Determine if this message leads to churn
            is_last_message = (msg_idx == num_messages - 1)
            label = 0

            if sender_role == "buyer":
                if will_churn and msg_idx >= num_messages - 3:
                    # Churn signals near the end
                    content = random.choice(self.buyer_messages['churn'])
                    if is_last_message:
                        label = 1  # This message leads to churn
                else:
                    msg_type = random.choice(['interested', 'positive'])
                    content = random.choice(self.buyer_messages[msg_type])
            else:
                if will_churn and msg_idx >= num_messages - 4:
                    # Seller might give negative response before churn
                    if random.random() < 0.4:
                        content = random.choice(self.seller_messages['negative'])
                    else:
                        content = random.choice(self.seller_messages['neutral'])
                else:
                    content = random.choice(self.seller_messages['positive'])

            # Time gap between messages
            if msg_idx == 0:
                dt_prev_sec = 0
            else:
                # Longer gaps before churn
                if will_churn and msg_idx >= num_messages - 2:
                    dt_prev_sec = random.randint(300, 7200)  # 5min ~ 2hours
                else:
                    dt_prev_sec = random.randint(10, 600)  # 10sec ~ 10min

            current_time = base_time + timedelta(seconds=dt_prev_sec)
            base_time = current_time

            # Generate reaction (more likely for positive messages)
            reaction = 1 if random.random() < 0.2 else 0

            messages.append({
                'message_id': f"{conv_id}_{msg_idx}",
                'conversation_id': conv_id,
                'sender_role': sender_role,
                'content': content,
                'created_at': current_time.isoformat(),
                'dt_prev_sec': dt_prev_sec,
                'msg_len': len(content),
                'reaction': reaction,
                'label': label
            })

        return messages

    def generate_dataset(self):
        """Generate complete dataset"""
        all_messages = []

        print(f"Generating {self.num_conversations} conversations...")
        for conv_id in range(self.num_conversations):
            if (conv_id + 1) % 100 == 0:
                print(f"  Generated {conv_id + 1}/{self.num_conversations} conversations")

            messages = self.generate_conversation(conv_id)
            all_messages.extend(messages)

        df = pd.DataFrame(all_messages)

        # Save to CSV
        output_path = self.config['data']['output_path']
        df.to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f"\nDataset saved to {output_path}")
        print(f"Total messages: {len(df)}")
        print(f"Churn messages: {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")

        return df


def main():
    generator = ChatDataGenerator()
    df = generator.generate_dataset()

    # Print sample
    print("\nSample messages:")
    print(df.head(10))


if __name__ == "__main__":
    main()
