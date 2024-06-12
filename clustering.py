import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import random
import pandas as pd
from openai import OpenAI
import json

model_name = 'gpt-4o'
personal_openai_key = 'YOUR_OPENAI_API_KEY_HERE'
client = OpenAI(api_key=personal_openai_key)

# Загружаем датасет со всеми отзывами
with open('all_responses_lines.txt', 'rt') as file:
    all_reviews_lines = file.read()
all_reviews = all_reviews_lines.split('\n')

# Получаем эмбеддинг для каждого отзыва
reviews_embeddings = []
for review in all_reviews:

    emb_api_response = client.embeddings.create(
        input=review,
        model="text-embedding-3-large"
    )
    embedding = emb_api_response.data[0].embedding

    reviews_embeddings.append((review, embedding))

# Кластеризуем и визуализируем эмбеддинги

# Set global random seed
random_seed = 13
np.random.seed(random_seed)
random.seed(random_seed)

# Extract embeddings
embeddings = np.array([item[1] for item in reviews_embeddings])

# Apply K-means clustering
num_clusters = 25  # Define the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=random_seed, n_init=13)
clusters = kmeans.fit_predict(embeddings)

# Apply t-SNE to reduce dimensionality to 2D
tsne = TSNE(n_components=2, random_state=random_seed)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot the 2D embeddings with cluster coloring
plt.figure(figsize=(12, 8))
for cluster_id in range(num_clusters):
    indices = np.where(clusters == cluster_id)
    plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], 
                label=f'Cluster {cluster_id}', 
                alpha=.5,
                color='grey'
    )
    # Calculate and plot centroid of the cluster
    centroid = embeddings_2d[indices].mean(axis=0)
    plt.text(centroid[0], centroid[1], str(cluster_id), fontsize=12, ha='center', va='center', color='black', weight='bold')

plt.title("2D Visualization of Embeddings with Clusters")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.grid(True)
plt.show()

# Теперь определим общую тему отзывов для каждого из 25 кластеров

# получаем все отзывы 
strings = [item[0] for item in response_embeddings]

for cluster_id in range(num_clusters):
    # Выводим ID кластера для дебага
    print(cluster_id)
    cluster_indices = np.where(clusters == cluster_id)[0]
    cluster_responses = [strings[i] for i in cluster_indices]
    cur_responses_lines = '\n'.join(cluster_responses)
    
    analyze_reviews_prompt = f'''
Твоя задача - проанализировать отзывы от клиентов компании по доставке еды и продуктов из магазинов и ресторанов. 
Компания имеет свое мобильное приложение и является агрегатором разных ресторанов и магазинов. 
Пользователь выбирает либо блюда из определенного ресторана, либо продукты питания из магазина, добавляет в корзину, оплачивает, указывает адрес доставки, и курьер привозит заказ прямо домой. 
Компания провела опрос среди своих клиентов и твоя задача проанализировать ответы из этой формы опроса. 
Все комментарии принадлежат к одной общей теме. Твоя задача понять, какая это тема. 
Прочитай внимательно все ответы из формы опроса и сформулируй тему, про которую писали пользователи. 
Ничего не выдумывай, опирайся только на предоставленные данные. 
Я дам тебе чаевые $200 за хороший анализ и выявление правильной темы, это очень важно для моей карьеры.

Каждый ответ на отдельной строке:
<ответы>
{cur_responses_lines}
</ответы>    
'''

    analyze_cluser_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": analyze_reviews_prompt}
        ]
    )
    analyze_cluser_text = analyze_cluser_response.choices[0].message.content

    # Выводим описание каждого из кластеров
    print(analyze_cluser_text)

# Теперь для каждого из отзывов определим, к какой из итоговых 18 категорий он относится 

window_size = 50
final_dict = dict()

for start_idx in range(0, len(all_reviews), window_size):
    end_idx = min(start_idx + window_size, len(all_reviews))

    current_reviews_lines = '\n'.join(all_reviews[start_idx:end_idx])
    categorize_reviews_prompt = f'''
Твоя задача - проанализировать отзывы от клиентов компании по доставке еды и продуктов из магазинов и ресторанов. 
Компания имеет свое мобильное приложение и является агрегатором разных ресторанов и магазинов. 
Пользователь выбирает либо блюда из определенного ресторана, либо продукты питания из магазина, добавляет в корзину, оплачивает, указывает адрес доставки, и курьер привозит заказ прямо домой. 
Компания провела опрос среди своих клиентов и твоя задача проанализировать ответы из этой формы опроса. 
Все комментарии делятся на несколько общих тем. Твоя задача для каждого ответа определить тему. 
Прочитай внимательно все ответы из формы опроса и для каждого из ответов напиши его тему. 
Входные данные - список возможных тем и список ответов, каждый ответ на отдельной строке. Ничего не выдумывай, опирайся только на предоставленные данные. 
Я дам тебе чаевые $200 за хороший анализ и правильное определение тем, это очень важно для моей карьеры.

<список тем>
1. Курьеры не могут найти адрес.
2. Курьеры звонят по телефону.
3. Долгое приготовление еды.
4. Долгая доставка.
5. Замороженные продукты приезжают растаявшими. 
6. Недоступны товары, которые отображаются доступными в приложении. 
7. Промокоды не срабатывали. 
8. Неправильная прожарка мяса. 
9. Несоблюдение просьб об исключении или добавлении ингредиентов.
10. Отсутствие необходимых столовых приборов.
11. Курьеры игнорируют описанные инструкции.
12. Фактическое время доставки значительно превышает обещанное.
13. Проблемы с производительностью и стабильностью мобильного приложения.
14. Курьеры не говорят на русском языке.
15. Положительные отзывы о приложении и качестве доставки.
16. Завышенные цены в сравнении с покупками напрямую.
17. Качество еды ухудшалось из-за долгой доставки - еда потеряла вкус или остыла.
18. Продукты приходят помятыми или в неудовлетворительном состоянии.
</список тем>

<ответы пользователей>
{current_reviews_lines}
</ответы пользователей>

Формат выходных данных - JSON. Например, 
"Курьер опоздал на полтора часа, неприятно.": "Долгая доставка.",
"Второй раз приезжает еда без салфеток, негоже так!": "Отсутствие необходимых столовых приборов.",
и так далее
    '''

    print(categorize_reviews_prompt)

    categorize_reviews_response = client.chat.completions.create(
        model=model_name,
        # Температуру ставим 0, чтобы модель меньше фантазировала, и точнее следовала инструкции
        temperature=0,
        messages=[
            {"role": "user", "content": categorize_reviews_prompt}
        ],
        # На выходе получим JSON - удобно парсить 
        response_format={"type": "json_object"}
    )
    categorize_reviews_text = categorize_reviews_response.choices[0].message.content

    print(categorize_reviews_text)

    # Записываем в общий словарь все пары (отзыв, категория отзыва)
    for key, value in json.loads(categorize_reviews_text).items():
        final_dict[key] = value

# На выходе у нас есть словарь final_dict в котором и хранится категория для каждого из отзывов

