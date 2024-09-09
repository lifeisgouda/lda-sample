import streamlit as st
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import preprocess_string
import networkx as nx
import plotly.graph_objs as go
import plotly.express as px

# 초기 문서
initial_docs = [
    "리뷰믿고 구매해봤어요 갠적으로 삼계탕은 누룽지 좋아해서 골랐어요 골랐는데 닭은 전체적으로 퍽퍽하고 생각이하로 맛 없어요 닭이 형체가 없고 뼈는 다 조각나서 굴러다니고 고기는 질기고.. 안먹을래요",
    "맛은 있는데 잔뼈가 너무 많음.. 먹기 좀 너무 힘듦.. 양도 그냥 금액에 맞춘듯한 느낌",
    "뭐지 전이랑 맛이 완전 다름 걸쭉했었는데 맹물같고 고기맛도 이상함",
    "포장도 깔끔하고 먹기도 간편하게 되어있네요 맛도 나쁘지 않고 가성비 괜찮은 것 같습니다 굿굿",
    "누룽지 구수한 맛이 정말 일품이에요. 아쉬운점은 뼈가 넘 잘익었는지 으스러져서 발라내기 힘드네용..",
    "반계탕이다보나 생각보다 양은 적어요 하지만 ㅁ 구수하니 누룽지향이 나서 맛은 좋아요. 밥 반그릇이랑 같이 먹으니 국물까지 싹쓸^_^"
]


@st.cache(allow_output_mutation=True)
def preprocess(text):
    return [token for token in preprocess_string(text) if len(token) > 1]


@st.cache(allow_output_mutation=True)
def create_lda_model(docs, num_topics):
    processed_docs = [preprocess(doc) for doc in docs]
    dictionary = corpora.Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=15)
    return lda_model, dictionary, corpus


def get_topic_terms(lda_model, num_words=10):
    return [dict(lda_model.show_topic(topicid, num_words)) for topicid in range(lda_model.num_topics)]


def create_word_topic_network(topic_terms):
    G = nx.Graph()
    for i, topic in enumerate(topic_terms):
        for word, _ in topic.items():
            G.add_edge(f"Topic {i + 1}", word)
    return G


st.title("Advanced LDA Interaction Dashboard")

# 세션 상태 초기화
if 'documents' not in st.session_state:
    st.session_state.documents = initial_docs

# 문서 입력
new_document = st.text_area("새 문서를 입력하세요:")
if st.button("문서 추가"):
    if new_document:
        st.session_state.documents.append(new_document)
        st.success("문서가 추가되었습니다.")

# 문서 목록 표시
st.write("현재 문서 목록:")
for i, doc in enumerate(st.session_state.documents):
    st.write(f"{i + 1}. {doc}")

# 토픽 수 선택
num_topics = st.slider("토픽 수", min_value=2, max_value=10, value=3)

if st.button("LDA 모델 실행"):
    lda_model, dictionary, corpus = create_lda_model(st.session_state.documents, num_topics)
    topic_terms = get_topic_terms(lda_model)

    # 단어-토픽 네트워크 생성
    G = create_word_topic_network(topic_terms)
    pos = nx.spring_layout(G)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x, node_y = [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        ),
        text=[node for node in G.nodes()],
        textposition="top center"
    )

    fig_network = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(
                                title='단어-토픽 네트워크',
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20, l=5, r=5, t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                            )

    st.plotly_chart(fig_network)

    # 토픽 진화 그래프 생성
    topic_evolution = []
    for i, doc in enumerate(st.session_state.documents):
        bow = dictionary.doc2bow(preprocess(doc))
        topic_dist = lda_model.get_document_topics(bow)
        for topic_id, prob in topic_dist:
            topic_evolution.append({
                'Document': i + 1,
                'Topic': f'Topic {topic_id + 1}',
                'Probability': prob
            })

    df_evolution = pd.DataFrame(topic_evolution)
    fig_evolution = px.line(df_evolution, x='Document', y='Probability', color='Topic',
                            title='문서에 따른 토픽 확률 변화')

    st.plotly_chart(fig_evolution)

st.write("이 대시보드는 LDA 토픽 모델링을 시각화합니다. 새 문서를 추가하고 토픽 수를 조정한 후 'LDA 모델 실행' 버튼을 클릭하여 결과를 확인하세요.")
