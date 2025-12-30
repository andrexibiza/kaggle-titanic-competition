import pandas as pd
import numpy as np

def analyze_wcg():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    data = pd.concat([train, test], sort=False)
    
    # Feature Engineering
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4}
    # Simplified mapping for WCG focus
    data['Title_Code'] = data['Title'].map(title_mapping).fillna(5) 
    
    data['Surname'] = data['Name'].apply(lambda x: x.split(',')[0].strip())
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    
    # Group Definitions
    data['Group_Surname'] = data['Surname']
    data['Group_Ticket'] = data['Ticket'].astype(str)
    
    # Analyze Ticket Groups
    print("--- Analysis of Ticket-based Groups ---")
    ticket_counts = data['Ticket'].value_counts()
    
    # Filter for groups > 1
    shared_tickets = ticket_counts[ticket_counts > 1].index
    data_shared = data[data['Ticket'].isin(shared_tickets)]
    
    print(f"Passengers with shared tickets: {len(data_shared)}")
    
    # Check survival consistency in shared tickets (Train set only)
    train_shared = data_shared[data_shared['Survived'].notnull()]
    
    ticket_consistency = train_shared.groupby('Ticket')['Survived'].agg(['mean', 'count', 'std'])
    print("\nTicket Consistency (Train Only):")
    print(ticket_consistency.sort_values('count', ascending=False).head(10))
    
    consistent_tickets = ticket_consistency[(ticket_consistency['mean'] == 1.0) | (ticket_consistency['mean'] == 0.0)]
    print(f"\nPerfectly consistent tickets in Train: {len(consistent_tickets)} out of {len(ticket_consistency)}")
    
    # Analyze WCG candidates in Test
    test_df = data[len(train):]
    women_children_test = test_df[(test_df['Title_Code'].isin([2,3,4]))]
    print(f"\nWomen/Children in Test: {len(women_children_test)}")
    
    # Check connections
    connected_by_ticket = 0
    connected_by_surname = 0
    
    for idx, row in women_children_test.iterrows():
        # Check Ticket in Train
        ticket = row['Ticket']
        if ticket in train['Ticket'].values:
            connected_by_ticket += 1
            
        # Check Surname in Train
        surname = row['Surname']
        if surname in train['Name'].apply(lambda x: x.split(',')[0].strip()).values:
            connected_by_surname += 1
            
    print(f"Test W/C connected to Train by Ticket: {connected_by_ticket}")
    print(f"Test W/C connected to Train by Surname: {connected_by_surname}")

if __name__ == "__main__":
    analyze_wcg()
