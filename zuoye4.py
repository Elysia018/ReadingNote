import os
import re
import math
import random
from typing import List, Dict, Set, Tuple, Union, Optional
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class DocumentSummary:
    """文档摘要信息"""
    doc_id: str
    title: str
    highlights: List[str]
    score: float
    authority: float
    pagerank: float

class Document:
    """文档类，表示一个文本文档"""
    def __init__(self, doc_id: str, content: str):
        self.doc_id = doc_id
        self.content = content
        self.title = self._extract_title(content)
        self.terms = self._tokenize(content)
        self.term_frequencies = self._compute_term_frequencies()
        self.term_positions = self._compute_term_positions()  # 记录词项位置
        self.authority_score = random.uniform(0.5, 1.0)  # 模拟权威性分数
        self.page_rank = random.uniform(0.1, 1.0)  # 模拟PageRank分数
    
    def _extract_title(self, content: str) -> str:
        """从内容中提取标题（第一行或文件名）"""
        first_line = content.split('\n')[0].strip()
        if len(first_line) < 100 and not first_line.islower():
            return first_line
        return os.path.splitext(self.doc_id)[0]
    
    def _tokenize(self, text: str) -> List[str]:
        """改进的分词函数，保留连字符和撇号"""
        text = text.lower()
        text = re.sub(r"[^\w\s'-]", '', text)
        return [term for term in text.split() if term]
    
    def _compute_term_frequencies(self) -> Dict[str, int]:
        """计算词项频率"""
        tf = defaultdict(int)
        for term in self.terms:
            tf[term] += 1
        return dict(tf)
    
    def _compute_term_positions(self) -> Dict[str, List[int]]:
        """记录每个词项出现的位置"""
        positions = defaultdict(list)
        for idx, term in enumerate(self.terms):
            positions[term].append(idx)
        return dict(positions)
    
    def contains_term(self, term: str) -> bool:
        """检查文档是否包含某个词项"""
        return term.lower() in self.term_frequencies
    
    def get_term_frequency(self, term: str) -> int:
        """获取词项在文档中的出现次数"""
        return self.term_frequencies.get(term.lower(), 0)
    
    def get_term_positions(self, term: str) -> List[int]:
        """获取词项在文档中的位置列表"""
        return self.term_positions.get(term.lower(), [])

class DocumentCollection:
    """文档集合类，管理所有文档"""
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.total_docs = 0
        self.term_statistics: Dict[str, int] = {}  # 包含词项的文档数
        self.term_document_frequencies: Dict[str, int] = {}  # 词项在所有文档中的出现次数
        self.doc_lengths: Dict[str, float] = {}  # 文档向量长度
        self.links: Dict[str, Set[str]] = defaultdict(set)  # 文档链接关系
    
    def add_document(self, doc_id: str, content: str):
        """添加文档到集合"""
        doc = Document(doc_id, content)
        self.documents[doc_id] = doc
        self.total_docs += 1
        
        # 更新词项统计信息
        for term in set(doc.terms):
            self.term_statistics[term] = self.term_statistics.get(term, 0) + 1
            self.term_document_frequencies[term] = self.term_document_frequencies.get(term, 0) + doc.get_term_frequency(term)
    
    def load_from_directory(self, directory: str):
        """从目录加载所有txt文件"""
        print(f"\nLoading documents from: {directory}")
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        loaded_files = 0
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                filepath = os.path.join(directory, filename)
                try:
                    for encoding in ['utf-8', 'utf-8-sig', 'gbk', 'latin-1']:
                        try:
                            with open(filepath, 'r', encoding=encoding) as f:
                                content = f.read()
                            self.add_document(filename, content)
                            loaded_files += 1
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        print(f"Warning: Could not decode {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
        
        # 模拟文档间的链接关系
        self._simulate_links()
        
        # 计算PageRank
        self._compute_pagerank()
        
        # 计算文档向量长度
        self._compute_document_lengths()
        
        print(f"Successfully loaded {loaded_files} documents")
        print(f"Total unique terms: {len(self.term_statistics)}")
    
    def _simulate_links(self):
        """模拟文档间的链接关系（实际应用中应从文档内容中提取）"""
        doc_ids = list(self.documents.keys())
        for doc_id in doc_ids:
            # 每个文档随机链接到3-5个其他文档
            num_links = random.randint(3, 5)
            linked_docs = random.sample(doc_ids, min(num_links, len(doc_ids)-1))
            if doc_id in linked_docs:
                linked_docs.remove(doc_id)  # 确保不自链
            self.links[doc_id].update(linked_docs)
    
    def _compute_pagerank(self, damping_factor=0.85, iterations=20):
        """计算PageRank分数"""
        print("\nComputing PageRank...")
        num_docs = len(self.documents)
        initial_value = 1.0 / num_docs
        ranks = {doc_id: initial_value for doc_id in self.documents}
        
        for _ in range(iterations):
            new_ranks = {}
            leak = 0.0
            
            # 处理没有出链的文档（泄漏）
            for doc_id in self.documents:
                if not self.links[doc_id]:
                    leak += ranks[doc_id] / num_docs
            
            # 计算新的PageRank
            for doc_id in self.documents:
                rank_sum = 0.0
                # 找出所有指向当前文档的链接
                for linking_doc, linked_docs in self.links.items():
                    if doc_id in linked_docs:
                        rank_sum += ranks[linking_doc] / len(linked_docs)
                
                new_ranks[doc_id] = (damping_factor * (rank_sum + leak)) + ((1 - damping_factor) / num_docs)
            
            # 更新所有文档的PageRank
            for doc_id in self.documents:
                ranks[doc_id] = new_ranks[doc_id]
                self.documents[doc_id].page_rank = ranks[doc_id]
    
    def _compute_document_lengths(self):
        """计算所有文档的向量长度（使用TF-IDF权重）"""
        for doc_id, doc in self.documents.items():
            length_squared = 0.0
            for term, tf in doc.term_frequencies.items():
                idf = self.inverse_document_frequency(term)
                weight = tf * idf
                length_squared += weight * weight
            self.doc_lengths[doc_id] = math.sqrt(length_squared)
    
    def inverse_document_frequency(self, term: str) -> float:
        """计算逆文档频率(IDF)"""
        if term not in self.term_statistics:
            return 0.0
        return math.log(self.total_docs / self.term_statistics[term])
    
    def get_document_ids(self) -> List[str]:
        return list(self.documents.keys())
    
    def show_term_stats(self, top_n=20):
        """显示词项统计"""
        if not self.term_statistics:
            print("No term statistics available")
            return
        
        sorted_terms = sorted(self.term_statistics.items(), 
                            key=lambda x: x[1], reverse=True)
        
        print("\nTop terms in collection:")
        for i, (term, count) in enumerate(sorted_terms[:top_n], 1):
            print(f"{i}. {term}: {count} occurrences")
    
    def show_sample_documents(self, n=3):
        """显示部分文档样本"""
        print("\nSample documents:")
        for i, (doc_id, doc) in enumerate(self.documents.items()):
            if i >= n:
                break
            print(f"\nDocument '{doc_id}':")
            print("Title:", doc.title)
            print("First 50 terms:", doc.terms[:50])
            print("Content preview:", doc.content[:100].replace('\n', ' ') + "...")
            print(f"Authority score: {doc.authority_score:.4f}, PageRank: {doc.page_rank:.4f}")

class BooleanRetriever:
    """布尔检索基类"""
    def __init__(self, collection: DocumentCollection):
        self.collection = collection
        # 虚拟同义词典（实际应用中应该更全面）
        self.synonyms = {
            "university": {"college", "institute", "academy"},
            "school": {"academy", "institution", "education"},
            "study": {"research", "investigation", "analysis"},
            "research": {"study", "investigation", "exploration"},
            "language": {"tongue", "speech", "dialect"},
            "computer": {"pc", "machine", "system"},
            "business": {"commerce", "trade", "industry"},
            "science": {"knowledge", "discipline", "field"},
            "art": {"craft", "skill", "creativity"},
            "culture": {"civilization", "society", "heritage"}
        }
    
    def search(self, query: str) -> List[str]:
        raise NotImplementedError("子类必须实现此方法")
    
    def _parse_query(self, query: str) -> List[Tuple[str, str]]:
        """改进的查询解析器，正确处理NOT操作符和括号"""
        # 预处理查询，将括号内的内容视为一个整体
        query = query.lower()
        tokens = []
        current_token = []
        in_parentheses = False
        
        for char in query:
            if char == '(':
                if current_token:
                    tokens.extend(''.join(current_token).split())
                    current_token = []
                in_parentheses = True
            elif char == ')':
                if current_token:
                    tokens.append(''.join(current_token))
                    current_token = []
                in_parentheses = False
            elif char.isspace() and not in_parentheses:
                if current_token:
                    tokens.append(''.join(current_token))
                    current_token = []
            else:
                current_token.append(char)
        
        if current_token:
            tokens.append(''.join(current_token))
        
        # 处理操作符
        terms = []
        i = 0
        n = len(tokens)
        
        while i < n:
            token = tokens[i].upper()
            if token == 'NOT':
                if i + 1 >= n:
                    raise ValueError(f"Missing term after operator: {token}")
                terms.append(('NOT', tokens[i+1].lower()))
                i += 2
            elif token in ('AND', 'OR'):
                if i + 1 >= n:
                    raise ValueError(f"Missing term after operator: {token}")
                terms.append((token, tokens[i+1].lower()))
                i += 2
            else:
                # 默认使用AND操作符
                if terms and terms[-1][0] not in ('AND', 'OR', 'NOT'):
                    terms.append(('AND', token.lower()))
                else:
                    terms.append(('AND', token.lower()))
                i += 1
        return terms
    
    def _expand_query_terms(self, terms: List[str]) -> List[str]:
        """查询扩展：添加同义词"""
        expanded_terms = set(terms)
        for term in terms:
            if term in self.synonyms:
                expanded_terms.update(self.synonyms[term])
        return list(expanded_terms)
    
    def _generate_summary(self, doc: Document, query_terms: List[str], max_snippets=3) -> DocumentSummary:
        """生成文档摘要，包含查询词项的高亮显示"""
        # 获取所有查询词项的位置
        term_positions = []
        for term in query_terms:
            term_positions.extend([(pos, term) for pos in doc.get_term_positions(term)])
        
        # 按位置排序
        term_positions.sort()
        
        # 生成高亮片段
        highlights = []
        last_pos = -10
        current_snippet = []
        
        for pos, term in term_positions:
            if pos - last_pos > 10:  # 如果位置间隔太大，开始新的片段
                if current_snippet:
                    highlights.append(self._create_snippet(doc.terms, current_snippet))
                    if len(highlights) >= max_snippets:
                        break
                    current_snippet = []
            current_snippet.append((pos, term))
            last_pos = pos
        
        if current_snippet and len(highlights) < max_snippets:
            highlights.append(self._create_snippet(doc.terms, current_snippet))
        
        # 如果没有找到高亮，使用文档开头
        if not highlights:
            highlights.append(" ".join(doc.terms[:50]) + "...")
        
        return DocumentSummary(
            doc_id=doc.doc_id,
            title=doc.title,
            highlights=highlights,
            score=0.0,  # 将在排序后填充
            authority=doc.authority_score,
            pagerank=doc.page_rank
        )
    
    def _create_snippet(self, terms: List[str], positions: List[Tuple[int, str]]) -> str:
        """创建高亮片段"""
        if not positions:
            return ""
        
        start_pos = max(0, positions[0][0] - 5)
        end_pos = min(len(terms), positions[-1][0] + 6)
        
        snippet = []
        for i in range(start_pos, end_pos):
            term = terms[i]
            # 检查是否是查询词项
            is_query_term = any(i == pos for pos, _ in positions)
            if is_query_term:
                snippet.append(f"**{term}**")  # 加粗高亮
            else:
                snippet.append(term)
        
        # 添加上下文指示
        if start_pos > 0:
            snippet.insert(0, "...")
        if end_pos < len(terms):
            snippet.append("...")
        
        return " ".join(snippet)
    
    def _jaccard_similarity(self, doc_terms: Set[str], query_terms: Set[str]) -> float:
        """计算Jaccard相似度系数"""
        intersection = len(doc_terms.intersection(query_terms))
        union = len(doc_terms.union(query_terms))
        return intersection / union if union > 0 else 0.0
    
    def _compute_tfidf_score(self, doc: Document, query_terms: List[str]) -> float:
        """计算TF-IDF分数"""
        score = 0.0
        for term in set(query_terms):
            tf = doc.get_term_frequency(term)
            idf = self.collection.inverse_document_frequency(term)
            score += tf * idf
        return score
    
    def _rank_results(self, doc_ids: List[str], query_terms: List[str], 
                     weights: Dict[str, float] = None) -> List[Tuple[str, float, DocumentSummary]]:
        """
        综合多种因素对结果进行排序并生成摘要
        权重参数示例: {'jaccard': 0.3, 'tfidf': 0.4, 'authority': 0.2, 'pagerank': 0.1}
        """
        if not doc_ids:
            return []
        
        # 默认权重
        if weights is None:
            weights = {'jaccard': 0.3, 'tfidf': 0.4, 'authority': 0.2, 'pagerank': 0.1}
        
        query_terms = [term.lower() for term in query_terms]
        query_terms_set = set(query_terms)
        
        ranked_results = []
        for doc_id in doc_ids:
            doc = self.collection.documents[doc_id]
            doc_terms_set = set(doc.terms)
            
            # 计算各项分数
            jaccard = self._jaccard_similarity(doc_terms_set, query_terms_set)
            tfidf = self._compute_tfidf_score(doc, query_terms)
            authority = doc.authority_score
            pagerank = doc.page_rank
            
            # 综合分数
            combined_score = (
                weights['jaccard'] * jaccard +
                weights['tfidf'] * tfidf +
                weights['authority'] * authority +
                weights['pagerank'] * pagerank
            )
            
            # 生成摘要
            summary = self._generate_summary(doc, query_terms)
            summary.score = combined_score  # 更新摘要中的分数
            
            ranked_results.append((doc_id, combined_score, summary))
        
        # 按综合分数降序排序
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        return ranked_results

class BruteForceRetriever(BooleanRetriever):
    """暴力检索实现"""
    def search(self, query: str) -> List[Tuple[str, float, DocumentSummary]]:
        try:
            terms = self._parse_query(query)
        except ValueError as e:
            print(f"Query parsing error: {str(e)}")
            return []
        
        if not terms:
            return []
        
        results = []
        query_terms = [term for op, term in terms]
        
        # 查询扩展
        expanded_terms = self._expand_query_terms(query_terms)
        print(f"Original query terms: {query_terms}")
        print(f"Expanded query terms: {expanded_terms}")
        
        for doc_id, doc in self.collection.documents.items():
            if self._matches(doc, terms):
                results.append(doc_id)
        
        # 对结果进行排序并生成摘要
        ranked_results = self._rank_results(results, expanded_terms)
        return ranked_results
    
    def _matches(self, doc: Document, terms: List[Tuple[str, str]]) -> bool:
        if not terms:
            return False
            
        first_op, first_term = terms[0]
        if first_op == 'NOT':
            result = not doc.contains_term(first_term)
        else:
            result = doc.contains_term(first_term)
        
        for op, term in terms[1:]:
            if op == 'AND':
                result = result and doc.contains_term(term)
            elif op == 'OR':
                result = result or doc.contains_term(term)
            elif op == 'NOT':
                result = result and not doc.contains_term(term)
        
        return result

class InvertedIndexRetriever(BooleanRetriever):
    """基于倒排索引的检索实现"""
    def __init__(self, collection: DocumentCollection):
        super().__init__(collection)
        self.inverted_index: Dict[str, Set[str]] = {}
        self._build_index()
    
    def _build_index(self):
        """构建倒排索引"""
        print("\nBuilding inverted index...")
        for doc_id, doc in self.collection.documents.items():
            for term in set(doc.terms):
                if term not in self.inverted_index:
                    self.inverted_index[term] = set()
                self.inverted_index[term].add(doc_id)
        print(f"Index built with {len(self.inverted_index)} unique terms")
    
    def search(self, query: str) -> List[Tuple[str, float, DocumentSummary]]:
        try:
            terms = self._parse_query(query)
        except ValueError as e:
            print(f"Query parsing error: {str(e)}")
            return []
        
        if not terms:
            return []
        
        result = None
        query_terms = [term for op, term in terms]
        
        # 查询扩展
        expanded_terms = self._expand_query_terms(query_terms)
        print(f"Original query terms: {query_terms}")
        print(f"Expanded query terms: {expanded_terms}")
        
        for op, term in terms:
            docs = self.inverted_index.get(term, set())
            
            if op == 'AND':
                if result is None:
                    result = docs.copy()
                else:
                    result.intersection_update(docs)
            elif op == 'OR':
                if result is None:
                    result = docs.copy()
                else:
                    result.update(docs)
            elif op == 'NOT':
                all_docs = set(self.collection.documents.keys())
                if result is None:
                    result = all_docs - docs
                else:
                    result.difference_update(docs)
        
        # 对结果进行排序并生成摘要
        ranked_results = self._rank_results(list(result) if result else [], expanded_terms)
        return ranked_results
    
    def check_term(self, term: str):
        """检查词项是否在索引中"""
        term = term.lower()
        if term in self.inverted_index:
            print(f"Term '{term}' found in {len(self.inverted_index[term])} documents")
            print("Sample documents:", list(self.inverted_index[term])[:3])
        else:
            print(f"Term '{term}' not found in index")

class SearchEngine:
    """搜索引擎主类"""
    def __init__(self, data_dir: str):
        self.collection = DocumentCollection()
        self.collection.load_from_directory(data_dir)
        self.brute_force_retriever = BruteForceRetriever(self.collection)
        self.inverted_index_retriever = InvertedIndexRetriever(self.collection)
        self.ranking_weights = {
            'default': {'jaccard': 0.3, 'tfidf': 0.4, 'authority': 0.2, 'pagerank': 0.1},
            'precision': {'jaccard': 0.2, 'tfidf': 0.5, 'authority': 0.2, 'pagerank': 0.1},
            'recall': {'jaccard': 0.4, 'tfidf': 0.3, 'authority': 0.2, 'pagerank': 0.1},
            'popularity': {'jaccard': 0.2, 'tfidf': 0.3, 'authority': 0.3, 'pagerank': 0.2}
        }
    
    def search(self, query: str, method: str = 'inverted', ranking_profile: str = 'default', 
               max_results: int = 10) -> List[DocumentSummary]:
        """
        执行搜索并返回带有摘要的结果
        :param query: 查询字符串
        :param method: 检索方法 ('brute' 或 'inverted')
        :param ranking_profile: 排序配置 ('default', 'precision', 'recall', 'popularity')
        :param max_results: 返回的最大结果数
        :return: 文档摘要列表
        """
        if method == 'brute':
            print(f"\nExecuting brute-force search for: {query}")
            results = self.brute_force_retriever.search(query)
        else:
            print(f"\nExecuting inverted-index search for: {query}")
            results = self.inverted_index_retriever.search(query)
        
        # 应用排序配置
        weights = self.ranking_weights.get(ranking_profile, self.ranking_weights['default'])
        
        # 显示搜索结果
        if results:
            print(f"\nTop {min(len(results), max_results)} results (ranking profile: {ranking_profile}):")
            for i, (doc_id, score, summary) in enumerate(results[:max_results], 1):
                print(f"\n#{i} {summary.title} (Score: {score:.4f})")
                print(f"Document ID: {doc_id}")
                print(f"Authority: {summary.authority:.2f}, PageRank: {summary.pagerank:.2f}")
                print("Highlights:")
                for j, highlight in enumerate(summary.highlights, 1):
                    print(f"  {j}. {highlight}")
        
        return [summary for _, _, summary in results[:max_results]]
    
    def analyze_query(self, query: str):
        """分析查询中的词项"""
        terms = self.inverted_index_retriever._parse_query(query)
        print("\nQuery analysis:")
        for op, term in terms:
            print(f"Operator: {op}, Term: '{term}'")
            self.inverted_index_retriever.check_term(term)
        
        # 显示查询扩展信息
        query_terms = [term for op, term in terms]
        expanded_terms = self.inverted_index_retriever._expand_query_terms(query_terms)
        if expanded_terms != query_terms:
            print("\nQuery expansion applied:")
            print(f"Original terms: {query_terms}")
            print(f"Expanded terms: {expanded_terms}")

if __name__ == "__main__":
    # 设置语料库目录路径
    data_dir = "D:\\信息检索\\语料"  # 替换为您的实际路径
    
    try:
        engine = SearchEngine(data_dir)
    except Exception as e:
        print(f"Failed to initialize search engine: {str(e)}")
        exit(1)
    
    # 显示统计信息
    engine.collection.show_term_stats()
    engine.collection.show_sample_documents()
    
    # 测试查询 - 基于您的语料库内容调整
    test_queries = [
        "language AND university",
        "art AND culture",
        "school AND NOT english",
        "environment OR studies",
        "business AND (school OR college)"
    ]
    
    ranking_profiles = ['default', 'precision', 'recall', 'popularity']
    
    for query in test_queries:
        print("\n" + "="*80)
        print(f"Testing query: {query}")
        
        # 分析查询词项
        engine.analyze_query(query)
        
        # 测试不同排序配置
        for profile in ranking_profiles:
            print(f"\n=== Ranking with profile: {profile} ===")
            
            # 使用倒排索引检索并显示摘要
            results = engine.search(query, method='inverted', ranking_profile=profile, max_results=3)

    # 交互式搜索
    print("\n" + "="*80)
    print("Interactive search mode (type 'exit' to quit)")
    while True:
        query = input("\nEnter your search query: ").strip()
        if query.lower() == 'exit':
            break
        
        if not query:
            continue
        
        print("\nSearch options:")
        print("1. Default ranking")
        print("2. Precision-focused")
        print("3. Recall-focused")
        print("4. Popularity-focused")
        choice = input("Choose ranking option (1-4, default=1): ").strip()
        
        profile = 'default'
        if choice == '2':
            profile = 'precision'
        elif choice == '3':
            profile = 'recall'
        elif choice == '4':
            profile = 'popularity'
        
        # 执行搜索
        results = engine.search(query, ranking_profile=profile, max_results=5)
        
        if not results:
            print("\nNo results found for your query.")