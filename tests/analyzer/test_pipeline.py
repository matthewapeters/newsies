"""
Test suite for Newsies Analyzer Pipeline
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime

from newsies_analyzer.pipelines.analyze import analyze_pipeline


class TestAnalyzerPipeline:
    """Test cases for content analysis pipeline"""

    @pytest.fixture
    def mock_redis_task_status(self):
        """Mock Redis task status for testing"""
        with patch('newsies_analyzer.pipelines.analyze.redis_task_status') as mock:
            yield mock

    @pytest.fixture
    def mock_summarization(self):
        """Mock text summarization functionality"""
        with patch('newsies_analyzer.pipelines.analyze.summarize_articles') as mock:
            yield mock

    @pytest.fixture
    def mock_ner_extraction(self):
        """Mock named entity recognition"""
        with patch('newsies_analyzer.pipelines.analyze.extract_named_entities') as mock:
            yield mock

    @pytest.fixture
    def mock_ngram_analysis(self):
        """Mock n-gram analysis functionality"""
        with patch('newsies_analyzer.pipelines.analyze.analyze_ngrams') as mock:
            yield mock

    @pytest.fixture
    def mock_embedding_generation(self):
        """Mock embedding generation"""
        with patch('newsies_analyzer.pipelines.analyze.generate_embeddings') as mock:
            yield mock

    def test_analyze_pipeline_success(self, mock_redis_task_status, 
                                    mock_summarization, mock_ner_extraction,
                                    mock_ngram_analysis, mock_embedding_generation):
        """Test successful execution of analyze pipeline"""
        task_id = "test_analyzer_task_123"
        
        # Mock successful execution of all steps
        mock_summarization.return_value = {"summaries": ["Summary 1", "Summary 2"]}
        mock_ner_extraction.return_value = {"entities": ["Person", "Organization"]}
        mock_ngram_analysis.return_value = {"ngrams": ["news", "article", "analysis"]}
        mock_embedding_generation.return_value = {"embeddings": [0.1, 0.2, 0.3]}
        
        # Execute pipeline
        analyze_pipeline(task_id)
        
        # Verify all steps were called
        mock_summarization.assert_called_once()
        mock_ner_extraction.assert_called_once()
        mock_ngram_analysis.assert_called_once()
        mock_embedding_generation.assert_called_once()
        
        # Verify task status updates
        assert mock_redis_task_status.set_status.call_count >= 4

    def test_analyze_pipeline_failure_at_summarization(self, mock_redis_task_status, mock_summarization):
        """Test pipeline failure at summarization step"""
        task_id = "test_analyzer_task_fail"
        
        # Mock failure at summarization step
        mock_summarization.side_effect = Exception("Summarization failed")
        
        with pytest.raises(Exception) as exc_info:
            analyze_pipeline(task_id)
        
        assert "Summarization failed" in str(exc_info.value)
        
        # Verify error status was set
        error_calls = [call for call in mock_redis_task_status.set_status.call_args_list 
                      if "error" in str(call)]
        assert len(error_calls) > 0

    def test_pipeline_status_updates(self, mock_redis_task_status, mock_summarization,
                                   mock_ner_extraction, mock_ngram_analysis, 
                                   mock_embedding_generation):
        """Test that pipeline updates task status at each step"""
        task_id = "test_status_updates"
        
        # Mock successful execution
        mock_summarization.return_value = {"summaries": []}
        mock_ner_extraction.return_value = {"entities": []}
        mock_ngram_analysis.return_value = {"ngrams": []}
        mock_embedding_generation.return_value = {"embeddings": []}
        
        analyze_pipeline(task_id)
        
        # Verify status updates for each step
        status_calls = mock_redis_task_status.set_status.call_args_list
        status_messages = [call[0][1] for call in status_calls]
        
        expected_statuses = [
            "started",
            "running - step: summarizing articles",
            "running - step: extracting named entities",
            "running - step: analyzing n-grams",
            "running - step: generating embeddings",
            "complete"
        ]
        
        for expected_status in expected_statuses:
            assert any(expected_status in status for status in status_messages)


class TestTextSummarization:
    """Test text summarization functionality"""

    def test_summarize_single_article(self):
        """Test summarization of a single article"""
        from newsies_analyzer.nlp.summarization import summarize_text
        
        long_text = """
        This is a long article about artificial intelligence and machine learning.
        The field has seen tremendous growth in recent years with advances in deep learning.
        Neural networks have become more sophisticated and capable of handling complex tasks.
        Applications range from natural language processing to computer vision.
        The future of AI looks promising with continued research and development.
        """
        
        with patch('newsies_analyzer.nlp.summarization.pipeline') as mock_pipeline:
            mock_pipeline.return_value = [{"summary_text": "AI field has grown with neural networks."}]
            
            summary = summarize_text(long_text, max_length=50)
            
            assert len(summary) > 0
            assert len(summary) < len(long_text)
            assert "AI" in summary or "neural" in summary

    def test_summarize_empty_text(self):
        """Test summarization of empty text"""
        from newsies_analyzer.nlp.summarization import summarize_text
        
        result = summarize_text("")
        assert result == "" or result is None

    def test_summarize_batch_articles(self):
        """Test batch summarization of multiple articles"""
        from newsies_analyzer.nlp.summarization import summarize_batch
        
        articles = [
            {"title": "Article 1", "content": "Content 1 " * 50},
            {"title": "Article 2", "content": "Content 2 " * 50},
            {"title": "Article 3", "content": "Content 3 " * 50}
        ]
        
        with patch('newsies_analyzer.nlp.summarization.summarize_text') as mock_summarize:
            mock_summarize.return_value = "Summary"
            
            results = summarize_batch(articles)
            
            assert len(results) == 3
            assert all("summary" in result for result in results)


class TestNamedEntityRecognition:
    """Test named entity recognition functionality"""

    def test_extract_entities_from_text(self):
        """Test entity extraction from text"""
        from newsies_analyzer.nlp.ner import extract_entities
        
        text = "Apple Inc. is headquartered in Cupertino, California. Tim Cook is the CEO."
        
        with patch('newsies_analyzer.nlp.ner.nlp') as mock_nlp:
            # Mock spaCy NLP pipeline
            mock_doc = Mock()
            mock_entities = [
                Mock(text="Apple Inc.", label_="ORG"),
                Mock(text="Cupertino", label_="GPE"),
                Mock(text="California", label_="GPE"),
                Mock(text="Tim Cook", label_="PERSON")
            ]
            mock_doc.ents = mock_entities
            mock_nlp.return_value = mock_doc
            
            entities = extract_entities(text)
            
            assert len(entities) == 4
            assert any(entity["text"] == "Apple Inc." for entity in entities)
            assert any(entity["label"] == "PERSON" for entity in entities)

    def test_extract_entities_empty_text(self):
        """Test entity extraction from empty text"""
        from newsies_analyzer.nlp.ner import extract_entities
        
        entities = extract_entities("")
        assert entities == []

    def test_filter_entities_by_type(self):
        """Test filtering entities by type"""
        from newsies_analyzer.nlp.ner import filter_entities_by_type
        
        entities = [
            {"text": "Apple Inc.", "label": "ORG"},
            {"text": "Tim Cook", "label": "PERSON"},
            {"text": "California", "label": "GPE"}
        ]
        
        persons = filter_entities_by_type(entities, "PERSON")
        assert len(persons) == 1
        assert persons[0]["text"] == "Tim Cook"


class TestNGramAnalysis:
    """Test n-gram analysis functionality"""

    def test_extract_bigrams(self):
        """Test bigram extraction"""
        from newsies_analyzer.nlp.ngrams import extract_ngrams
        
        text = "artificial intelligence machine learning deep learning neural networks"
        
        bigrams = extract_ngrams(text, n=2)
        
        assert len(bigrams) > 0
        assert any("artificial intelligence" in bigram for bigram in bigrams)
        assert any("machine learning" in bigram for bigram in bigrams)

    def test_extract_trigrams(self):
        """Test trigram extraction"""
        from newsies_analyzer.nlp.ngrams import extract_ngrams
        
        text = "natural language processing artificial intelligence machine learning"
        
        trigrams = extract_ngrams(text, n=3)
        
        assert len(trigrams) > 0
        assert any("natural language processing" in trigram for trigram in trigrams)

    def test_ngram_frequency_analysis(self):
        """Test n-gram frequency analysis"""
        from newsies_analyzer.nlp.ngrams import analyze_ngram_frequency
        
        texts = [
            "machine learning artificial intelligence",
            "deep learning machine learning",
            "artificial intelligence neural networks"
        ]
        
        frequency_analysis = analyze_ngram_frequency(texts, n=2)
        
        assert "machine learning" in frequency_analysis
        assert frequency_analysis["machine learning"] >= 2


class TestEmbeddingGeneration:
    """Test embedding generation functionality"""

    def test_generate_text_embeddings(self):
        """Test text embedding generation"""
        from newsies_analyzer.nlp.embeddings import generate_embeddings
        
        texts = [
            "This is a test sentence about artificial intelligence.",
            "Machine learning is a subset of AI.",
            "Neural networks are used in deep learning."
        ]
        
        with patch('newsies_analyzer.nlp.embeddings.embedding_model') as mock_model:
            mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
            
            embeddings = generate_embeddings(texts)
            
            assert len(embeddings) == 3
            assert all(len(embedding) == 3 for embedding in embeddings)

    def test_similarity_calculation(self):
        """Test embedding similarity calculation"""
        from newsies_analyzer.nlp.embeddings import calculate_similarity
        
        embedding1 = [0.1, 0.2, 0.3]
        embedding2 = [0.4, 0.5, 0.6]
        
        similarity = calculate_similarity(embedding1, embedding2)
        
        assert 0 <= similarity <= 1
        assert isinstance(similarity, float)

    def test_batch_embedding_generation(self):
        """Test batch embedding generation"""
        from newsies_analyzer.nlp.embeddings import generate_batch_embeddings
        
        articles = [
            {"title": "AI News", "content": "Content about AI"},
            {"title": "ML Update", "content": "Content about ML"},
            {"title": "Tech News", "content": "Content about technology"}
        ]
        
        with patch('newsies_analyzer.nlp.embeddings.generate_embeddings') as mock_generate:
            mock_generate.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
            
            results = generate_batch_embeddings(articles)
            
            assert len(results) == 3
            assert all("embeddings" in result for result in results)


@pytest.mark.integration
class TestAnalyzerIntegration:
    """Integration tests for analyzer components"""

    def test_full_analysis_workflow(self):
        """Test complete analysis workflow"""
        sample_articles = [
            {
                "title": "AI Breakthrough in Healthcare",
                "content": "Artificial intelligence has made significant strides in medical diagnosis. Machine learning algorithms can now detect diseases with high accuracy."
            },
            {
                "title": "Climate Change Impact",
                "content": "Global warming continues to affect weather patterns worldwide. Scientists are studying the long-term effects on ecosystems."
            }
        ]
        
        # This would test the complete workflow with mocked external dependencies
        # For now, just verify the structure exists
        from newsies_analyzer.pipelines.analyze import process_articles_batch
        
        with patch('newsies_analyzer.pipelines.analyze.summarize_batch') as mock_summarize:
            with patch('newsies_analyzer.pipelines.analyze.extract_entities_batch') as mock_ner:
                with patch('newsies_analyzer.pipelines.analyze.analyze_ngrams_batch') as mock_ngrams:
                    mock_summarize.return_value = [{"summary": "AI summary"}]
                    mock_ner.return_value = [{"entities": ["AI", "Healthcare"]}]
                    mock_ngrams.return_value = [{"ngrams": ["artificial intelligence"]}]
                    
                    results = process_articles_batch(sample_articles)
                    
                    assert len(results) == 2
                    assert all("analysis" in result for result in results)


@pytest.mark.unit
class TestAnalyzerUtilities:
    """Test utility functions in analyzer package"""

    def test_text_preprocessing(self):
        """Test text preprocessing utilities"""
        from newsies_analyzer.utils.text_processing import preprocess_text
        
        dirty_text = "  This is a TEST with\n\nextra whitespace and CAPS!  "
        clean_text = preprocess_text(dirty_text)
        
        assert clean_text == "this is a test with extra whitespace and caps"
        assert "\n" not in clean_text
        assert not clean_text.startswith(" ")

    def test_language_detection(self):
        """Test language detection"""
        from newsies_analyzer.utils.language import detect_language
        
        english_text = "This is an English sentence."
        spanish_text = "Esta es una oración en español."
        
        assert detect_language(english_text) == "en"
        assert detect_language(spanish_text) == "es"

    def test_content_filtering(self):
        """Test content filtering utilities"""
        from newsies_analyzer.utils.filtering import filter_content
        
        content = "This article contains relevant information about technology and some spam content."
        
        filtered = filter_content(content, min_length=10, remove_spam=True)
        
        assert len(filtered) >= 10
        assert "relevant information" in filtered
