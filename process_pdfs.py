import os
import pandas as pd
import time
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from term_counter import load_lexicon, process_pdf_for_terms, MatchClassifier
from pdf_highlighter import highlight_terms_in_pdf

def format_time(seconds):
    """Format time duration in a human-readable format."""
    return str(timedelta(seconds=round(seconds)))

def create_category_distribution_chart(metrics, output_dir):
    """Create a bar chart showing the distribution of innovation categories."""
    plt.figure(figsize=(12, 6))
    stats = metrics['category_stats']
    
    # Create bar chart
    ax = sns.barplot(x=stats.index, y='percentage_of_files', data=stats)
    
    # Customize the chart
    plt.title('Distribution of Innovation Categories Across Files', pad=20)
    plt.xlabel('Innovation Category')
    plt.ylabel('Percentage of Files (%)')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for i, v in enumerate(stats['percentage_of_files']):
        ax.text(i, v, f'{v}%', ha='center', va='bottom')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_distribution.png'))
    plt.close()

def create_co_occurrence_heatmap(metrics, output_dir):
    """Create a heatmap showing category co-occurrence."""
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(metrics['category_co_occurrence'], 
                annot=True, 
                fmt='d',
                cmap='YlOrRd',
                cbar_kws={'label': 'Co-occurrence Count'})
    
    plt.title('Innovation Category Co-occurrence Matrix', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_co_occurrence.png'))
    plt.close()

def create_innovation_complexity_pie(metrics, output_dir):
    """Create a pie chart showing the distribution of innovation complexity."""
    plt.figure(figsize=(10, 8))
    
    dist = metrics['category_distribution']
    labels = [idx.replace('_', ' ').capitalize() for idx in dist.index]
    
    plt.pie(dist['percentage'], labels=labels, autopct='%1.1f%%',
            colors=sns.color_palette('pastel'))
    plt.title('Distribution of Innovation Complexity\n(Number of Categories per File)')
    
    plt.savefig(os.path.join(output_dir, 'innovation_complexity.png'))
    plt.close()

def create_term_type_distribution(classifier, output_dir):
    """Create a stacked bar chart showing primary vs related term distribution."""
    stats = classifier.get_category_statistics()
    
    plt.figure(figsize=(12, 6))
    
    # Prepare data for stacked bar chart
    primary = stats['primary_term_matches']
    related = stats['related_term_matches']
    
    # Create stacked bar chart
    ax = plt.gca()
    ax.bar(stats['category'], primary, label='Primary Terms')
    ax.bar(stats['category'], related, bottom=primary, label='Related Terms')
    
    plt.title('Distribution of Primary vs Related Terms by Category', pad=20)
    plt.xlabel('Innovation Category')
    plt.ylabel('Number of Matches')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    
    # Add value labels
    for i in range(len(stats)):
        # Label for primary terms
        if primary[i] > 0:
            plt.text(i, primary[i]/2, str(primary[i]), ha='center', va='center')
        # Label for related terms
        if related[i] > 0:
            plt.text(i, primary[i] + related[i]/2, str(related[i]), ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'term_type_distribution.png'))
    plt.close()

def create_visualizations(metrics, classifier, output_dir):
    """Create and save all visualization charts."""
    print("Generating visualization charts...")
    
    # Set the style for all plots
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Create individual charts
    create_category_distribution_chart(metrics, output_dir)
    create_co_occurrence_heatmap(metrics, output_dir)
    create_innovation_complexity_pie(metrics, output_dir)
    create_term_type_distribution(classifier, output_dir)

def save_classification_results(classifier, output_dir):
    """Save detailed classification results to CSV files."""
    # Save category statistics
    stats_df = classifier.get_category_statistics()
    stats_df.to_csv(os.path.join(output_dir, "detailed_category_statistics.csv"))
    
    # Save innovation patterns
    patterns_df = classifier.get_innovation_patterns()
    if not patterns_df.empty:
        patterns_df.to_csv(os.path.join(output_dir, "innovation_patterns.csv"), index=False)

def generate_innovation_metrics(results_df, total_files):
    """Generate detailed innovation metrics from the results."""
    # Files with any innovation
    files_with_innovation = len(results_df['filename'].unique())
    innovation_percentage = round((files_with_innovation / total_files * 100), 2)
    
    # Category-level statistics
    category_stats = results_df.groupby('category').agg({
        'filename': 'nunique',
        'matched_term': 'count'
    }).rename(columns={
        'filename': 'unique_files',
        'matched_term': 'total_matches'
    })
    
    # Calculate percentages
    category_stats['percentage_of_files'] = round((category_stats['unique_files'] / total_files * 100), 2)
    category_stats['matches_per_file'] = round((category_stats['total_matches'] / category_stats['unique_files']), 2)
    
    # Calculate co-occurrence matrix
    file_category_matrix = pd.crosstab(results_df['filename'], results_df['category'])
    co_occurrence = pd.crosstab(results_df['filename'], results_df['category']).gt(0).astype(int)
    category_co_occurrence = co_occurrence.T.dot(co_occurrence)
    
    # Distribution of innovation categories per file
    innovation_counts = co_occurrence.sum(axis=1)
    category_distribution = pd.Series({
        '1_category': (innovation_counts == 1).sum(),
        '2_categories': (innovation_counts == 2).sum(),
        '3_or_more_categories': (innovation_counts >= 3).sum()
    })
    category_distribution_pct = round((category_distribution / files_with_innovation * 100), 2)
    
    return {
        'overall_metrics': {
            'total_files': total_files,
            'files_with_innovation': files_with_innovation,
            'innovation_percentage': innovation_percentage,
            'avg_categories_per_file': round(innovation_counts.mean(), 2)
        },
        'category_stats': category_stats,
        'category_co_occurrence': category_co_occurrence,
        'category_distribution': pd.DataFrame({
            'count': category_distribution,
            'percentage': category_distribution_pct
        })
    }

def save_innovation_metrics(metrics, output_dir):
    """Save innovation metrics to separate CSV files."""
    # Save overall metrics
    pd.DataFrame([metrics['overall_metrics']]).to_csv(
        os.path.join(output_dir, "overall_metrics.csv")
    )
    
    # Save category statistics
    metrics['category_stats'].to_csv(
        os.path.join(output_dir, "category_statistics.csv")
    )
    
    # Save co-occurrence matrix
    metrics['category_co_occurrence'].to_csv(
        os.path.join(output_dir, "category_co_occurrence.csv")
    )
    
    # Save category distribution
    metrics['category_distribution'].to_csv(
        os.path.join(output_dir, "category_distribution.csv")
    )

def process_all_pdfs(pdf_folder, lexicon_file, output_folder, threshold=85):
    """Process all PDFs for term counting, context, and highlighting."""
    start_time = time.time()
    
    # Create timestamped subdirectory
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    timestamped_output_dir = os.path.join(output_folder, timestamp)
    os.makedirs(timestamped_output_dir, exist_ok=True)

    print(f"\nLoading lexicon from {lexicon_file}...")
    lexicon_load_start = time.time()
    lexicon_terms = load_lexicon(lexicon_file)
    print(f"Loaded {len(lexicon_terms)} lexicon terms in {format_time(time.time() - lexicon_load_start)}")
    
    results = []
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    total_files = len(pdf_files)
    
    print(f"\nProcessing {total_files} PDF files...")
    
    for file_num, filename in enumerate(pdf_files, 1):
        file_start_time = time.time()
        pdf_path = os.path.join(pdf_folder, filename)
        print(f"\nProcessing file {file_num}/{total_files}: {filename}")

        # Count terms and track context
        matches = process_pdf_for_terms(pdf_path, lexicon_terms, threshold)
        
        # Only process and extend results if there are matches
        if matches:
            # Add filename to each match
            for match in matches:
                match["filename"] = filename
            results.extend(matches)

            # Only create highlighted PDF if there are matches
            highlighted_pdf_path = os.path.join(timestamped_output_dir, f"highlighted_{filename}")
            highlight_terms_in_pdf(pdf_path, matches, highlighted_pdf_path)
            print(f"Created highlighted PDF with {len(matches)} matches")
        else:
            print(f"No matches found in {filename}")
        
        file_duration = time.time() - file_start_time
        print(f"File processing time: {format_time(file_duration)}")
    
    # Only create output files if there are any results
    if results:
        print("\nGenerating reports...")
        report_start_time = time.time()
        
        # Create classifier instance
        classifier = MatchClassifier(results)
        
        # Save detailed results to CSV
        results_df = pd.DataFrame(results)
        csv_output_path = os.path.join(timestamped_output_dir, "term_locations_with_context.csv")
        results_df.to_csv(csv_output_path, index=False)
        print(f"Term location results saved to: {csv_output_path}")

        # Generate and save innovation metrics
        print("Generating innovation metrics...")
        metrics = generate_innovation_metrics(results_df, total_files)
        save_innovation_metrics(metrics, timestamped_output_dir)
        
        # Save classification results
        save_classification_results(classifier, timestamped_output_dir)
        
        # Create visualizations with both metrics and classifier data
        create_visualizations(metrics, classifier, timestamped_output_dir)
        
        # Print summary of findings
        overall = metrics['overall_metrics']
        print(f"\nInnovation Analysis Summary:")
        print(f"- {overall['files_with_innovation']} out of {overall['total_files']} files ({overall['innovation_percentage']}%) contain innovative techniques")
        print(f"- Average number of innovation categories per file: {overall['avg_categories_per_file']}")
        
        # Distribution summary
        dist = metrics['category_distribution']
        print("\nDistribution of innovation categories:")
        for idx, row in dist.iterrows():
            category_name = idx.replace('_', ' ').capitalize()
            print(f"- {category_name}: {row['count']} files ({row['percentage']}%)")
        
        # Term type distribution summary
        stats = classifier.get_category_statistics()
        print("\nTerm type distribution:")
        for _, row in stats.iterrows():
            print(f"\n{row['category']}:")
            print(f"- Primary term matches: {row['primary_term_matches']}")
            print(f"- Related term matches: {row['related_term_matches']}")
            print(f"- Most common terms: {', '.join(list(row['most_common_terms'].keys())[:3])}")
        
        print(f"\nReport generation time: {format_time(time.time() - report_start_time)}")
        print("\nVisualization charts have been saved to the output directory:")
        print("- category_distribution.png")
        print("- category_co_occurrence.png")
        print("- innovation_complexity.png")
        print("- term_type_distribution.png")
    else:
        print("\nNo matches found in any files")
    
    total_duration = time.time() - start_time
    print(f"\nTotal processing time: {format_time(total_duration)}")
    print(f"Average time per file: {format_time(total_duration/total_files)}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python process_pdfs.py <pdf_folder> <lexicon_file> <output_folder>")
        sys.exit(1)

    pdf_folder = sys.argv[1]
    lexicon_file = sys.argv[2]
    output_folder = sys.argv[3]

    process_all_pdfs(pdf_folder, lexicon_file, output_folder)