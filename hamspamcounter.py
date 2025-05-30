import pandas as pd

def count_spam_ham():
    # Read the CSV file
    print("Reading the CSV file...")
    df = pd.read_csv('processed_data 2.csv')
    
    # Count the number of spam and ham messages
    spam_count = (df['label'] == 1).sum()
    ham_count = (df['label'] == 0).sum()
    
    # Print the results
    print("\nMessage Counts:")
    print(f"Spam messages: {spam_count}")
    print(f"Ham messages: {ham_count}")
    print(f"Total messages: {len(df)}")
    
    # Calculate percentages
    total = len(df)
    spam_percentage = (spam_count / total) * 100
    ham_percentage = (ham_count / total) * 100
    
    print("\nPercentages:")
    print(f"Spam: {spam_percentage:.2f}%")
    print(f"Ham: {ham_percentage:.2f}%")

if __name__ == "__main__":
    try:
        count_spam_ham()
    except Exception as e:
        print(f"An error occurred: {e}") 