"""
Example Usage of Legal RAG Hierarchical Chunker

This script demonstrates how to use the chunking system with different document types.
"""

import sys
sys.path.insert(0, '/home/claude')

from legal_rag_chunker import HierarchicalChunkingPipeline
from legal_rag_chunker.core import DocumentType


# Example 1: Employment Agreement with Numbered Sections
def example_numbered_sections():
    print("=" * 80)
    print("EXAMPLE 1: Employment Agreement (Numbered Sections)")
    print("=" * 80)
    
    employment_agreement = """
EMPLOYMENT AGREEMENT

This Employment Agreement is entered into as of January 1, 2024.

1. EMPLOYMENT TERMS

The Company hereby employs the Employee, and the Employee accepts employment with the Company, subject to the terms and conditions set forth in this Agreement.

1.1 Position and Duties

Employee shall serve as Senior Software Engineer. Employee shall report to the Chief Technology Officer and shall perform such duties as are customarily associated with such position.

1.2 Work Location

Employee's primary work location shall be the Company's headquarters in San Francisco, California. The Company may require Employee to travel on Company business.

2. COMPENSATION

Employee shall receive the following compensation for services rendered:

2.1 Base Salary

Employee shall receive a base salary of $150,000 per year, payable in accordance with the Company's standard payroll practices.

2.2 Bonus

Employee shall be eligible for an annual performance bonus of up to 20% of base salary, subject to achievement of performance goals.

2.3 Equity

Employee shall receive stock options as determined by the Board of Directors.

3. BENEFITS

3.1 Health Insurance

Employee shall be eligible to participate in the Company's health insurance plans.

3.2 Paid Time Off

Employee shall be entitled to 20 days of paid time off per year.

4. CONFIDENTIALITY

Employee acknowledges that during employment, Employee will have access to confidential information and trade secrets of the Company.
"""
    
    # Create pipeline
    pipeline = HierarchicalChunkingPipeline()
    
    # Process document
    result = pipeline.process(employment_agreement)
    
    # Display results
    print(f"\nDetected Document Type: {result.detection_result.doc_type.value}")
    print(f"Detection Confidence: {result.detection_result.confidence:.2f}")
    print(f"\nTotal Chunks: {result.statistics['total_chunks']}")
    print(f"Average Chunk Size: {result.statistics['avg_chunk_size']:.0f} characters")
    print(f"Max Hierarchy Depth: {result.statistics['max_hierarchy_depth']}")
    
    print("\n" + "-" * 80)
    print("CHUNKS:")
    print("-" * 80)
    
    for i, chunk in enumerate(result.chunks[:5], 1):  # Show first 5 chunks
        print(f"\n[Chunk {i}] {chunk.chunk_id}")
        print(f"Hierarchy: {' > '.join(chunk.hierarchy)}")
        print(f"Content ({len(chunk.content)} chars):")
        print(chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content)
    
    if len(result.chunks) > 5:
        print(f"\n... and {len(result.chunks) - 5} more chunks")


# Example 2: Legal Statute with Articles
def example_statute():
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: Legal Statute (Articles & Clauses)")
    print("=" * 80)
    
    statute_text = """
CONSUMER PROTECTION ACT

Article 1 - Definitions

For the purposes of this Act:
(1) "Consumer" means any natural person who purchases goods or services for personal use.
(2) "Supplier" means any person engaged in the business of supplying goods or services to consumers.
(3) "Unfair practice" means any practice that deceives or misleads consumers.

Article 2 - Consumer Rights

Section 1. Right to Information
(1) Every consumer has the right to complete and accurate information about goods and services.
(2) Suppliers must disclose all material facts about products, including:
    (a) Price and payment terms
    (b) Quality and safety characteristics
    (c) Warranty terms

Section 2. Right to Safety
Consumers have the right to protection against goods and services that are hazardous to health or safety.

Article 3 - Unfair Practices Prohibited

Section 1. False Advertising
(1) No supplier shall engage in false or misleading advertising.
(2) Advertising is false if it contains misrepresentations of material facts.

Section 2. Deceptive Pricing
Suppliers shall not engage in deceptive pricing practices, including:
(a) Bait and switch tactics
(b) False reference pricing
(c) Hidden fees

Article 4 - Remedies

Consumers who suffer loss due to unfair practices may seek the following remedies:
(1) Rescission of the contract
(2) Refund of purchase price
(3) Damages for losses suffered
(4) Injunctive relief
"""
    
    pipeline = HierarchicalChunkingPipeline()
    result = pipeline.process(statute_text)
    
    print(f"\nDetected Document Type: {result.detection_result.doc_type.value}")
    print(f"Detection Confidence: {result.detection_result.confidence:.2f}")
    print(f"\nTotal Chunks: {result.statistics['total_chunks']}")
    
    print("\n" + "-" * 80)
    print("CHUNKS:")
    print("-" * 80)
    
    for i, chunk in enumerate(result.chunks[:4], 1):
        print(f"\n[Chunk {i}] {chunk.chunk_id}")
        print(f"Hierarchy: {' > '.join(chunk.hierarchy)}")
        print(f"Type: {chunk.metadata.get('type')}")
        print(f"Content preview: {chunk.content[:150]}...")


# Example 3: Policy Document with Lists
def example_policy():
    print("\n\n" + "=" * 80)
    print("EXAMPLE 3: Privacy Policy (List-Heavy)")
    print("=" * 80)
    
    policy_text = """
PRIVACY POLICY

INFORMATION WE COLLECT

We collect several types of information from and about users of our services:

(a) Personal identification information, including but not limited to:
    • Name
    • Email address
    • Phone number
    • Mailing address

(b) Technical information, including:
    • IP address
    • Browser type
    • Operating system
    • Device information

(c) Usage information about how you use our services:
    • Pages visited
    • Time spent on pages
    • Links clicked
    • Search queries

HOW WE USE YOUR INFORMATION

We use the information we collect for various purposes:

(a) To provide and maintain our services
(b) To notify you about changes to our services
(c) To provide customer support
(d) To gather analysis or valuable information to improve our services
(e) To monitor the usage of our services
(f) To detect, prevent and address technical issues

SHARING YOUR INFORMATION

We may share your information in the following situations:

(a) With service providers who assist us in operating our services
(b) With business partners for joint marketing efforts
(c) In connection with a merger, sale, or acquisition
(d) To comply with legal obligations
(e) To protect our rights and safety

YOUR RIGHTS

You have certain rights regarding your personal information:

(a) Right to access your personal information
(b) Right to correct inaccurate information
(c) Right to delete your information
(d) Right to restrict processing
(e) Right to data portability
(f) Right to withdraw consent
"""
    
    pipeline = HierarchicalChunkingPipeline()
    result = pipeline.process(policy_text)
    
    print(f"\nDetected Document Type: {result.detection_result.doc_type.value}")
    print(f"Detection Confidence: {result.detection_result.confidence:.2f}")
    print(f"\nTotal Chunks: {result.statistics['total_chunks']}")
    print(f"Chunk Types: {result.statistics['chunk_types']}")
    
    print("\n" + "-" * 80)
    print("CHUNKS:")
    print("-" * 80)
    
    for i, chunk in enumerate(result.chunks[:3], 1):
        print(f"\n[Chunk {i}] {chunk.chunk_id}")
        print(f"Hierarchy: {' > '.join(chunk.hierarchy) if chunk.hierarchy else 'No hierarchy'}")
        print(f"Content preview: {chunk.content[:200]}...")


# Example 4: Custom Configuration
def example_custom_config():
    print("\n\n" + "=" * 80)
    print("EXAMPLE 4: Custom Configuration")
    print("=" * 80)
    
    text = """
1. INTRODUCTION

This is a sample document to demonstrate custom chunking configuration.

1.1 Purpose

The purpose of this document is to show how you can customize chunk sizes and other parameters.

2. MAIN CONTENT

This section contains the main content of the document with various subsections.

2.1 First Subsection

Content for the first subsection.

2.2 Second Subsection

Content for the second subsection.
"""
    
    # Custom configuration
    custom_config = {
        'max_chunk_size': 300,  # Smaller chunks
        'min_chunk_size': 50,
        'preserve_context': True,
    }
    
    pipeline = HierarchicalChunkingPipeline(chunker_config=custom_config)
    result = pipeline.process(text)
    
    print(f"\nUsing custom config: {custom_config}")
    print(f"\nTotal Chunks: {result.statistics['total_chunks']}")
    print(f"Average Chunk Size: {result.statistics['avg_chunk_size']:.0f} characters")
    
    for i, chunk in enumerate(result.chunks, 1):
        print(f"\n[Chunk {i}] {len(chunk.content)} chars")
        print(f"Hierarchy: {' > '.join(chunk.hierarchy)}")


# Example 5: Export Formats
def example_export():
    print("\n\n" + "=" * 80)
    print("EXAMPLE 5: Export to Different Formats")
    print("=" * 80)
    
    text = """
1. SECTION ONE
Content for section one.

2. SECTION TWO
Content for section two.
"""
    
    pipeline = HierarchicalChunkingPipeline()
    result = pipeline.process(text)
    
    print("\n--- JSON Export ---")
    json_output = pipeline.export_chunks(result.chunks, format='json', include_metadata=False)
    print(json_output[:300] + "...")
    
    print("\n--- CSV Export ---")
    csv_output = pipeline.export_chunks(result.chunks, format='csv', include_metadata=False)
    print(csv_output[:300] + "...")
    
    print("\n--- Markdown Export ---")
    md_output = pipeline.export_chunks(result.chunks, format='markdown', include_metadata=False)
    print(md_output[:300] + "...")


# Example 6: Batch Processing
def example_batch_processing():
    print("\n\n" + "=" * 80)
    print("EXAMPLE 6: Batch Processing Multiple Documents")
    print("=" * 80)
    
    documents = [
        {
            'id': 'doc1',
            'text': '1. SECTION\nContent here.'
        },
        {
            'id': 'doc2',
            'text': 'Article 1\nContent here.'
        },
        {
            'id': 'doc3',
            'text': 'This is unstructured text without clear sections.'
        }
    ]
    
    pipeline = HierarchicalChunkingPipeline()
    results = pipeline.process_batch(documents)
    
    print(f"\nProcessed {len(results)} documents")
    
    for i, result in enumerate(results, 1):
        print(f"\nDocument {i}:")
        print(f"  Type: {result.detection_result.doc_type.value}")
        print(f"  Chunks: {result.statistics['total_chunks']}")
        print(f"  Confidence: {result.detection_result.confidence:.2f}")


def main():
    """Run all examples"""
    
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "LEGAL RAG HIERARCHICAL CHUNKER EXAMPLES" + " " * 24 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Run examples
    example_numbered_sections()
    example_statute()
    example_policy()
    example_custom_config()
    example_export()
    example_batch_processing()
    
    print("\n\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
