import numpy as np
from itertools import permutations

# Starting parameters
print()
investment = 10000  # Starting investment in GBP
r = 0.0430 # UK bond rate, which is continuously compounded. 

# Opportunity 1

Exchange_Matrix = np.matrix([
                             [1.0000,1.2282,1.2741,1.8235],
                             [0.8142,1.0000,1.0373,1.4845],
                             [0.7848,0.9640,1.0000,1.4309],
                             [0.5484,0.6736,0.6989,1.0000]
                             ])

currencies = ['GBP', 'EUR', 'USD', 'CAD'] # Follows the same index of the Exchange_Matrix, using this gives quick and easy access through the matrix to calculate exchange rates

def exchange_gbp(path: list) -> float:
    '''
    Path: A path can take the form ['GBP', 'EUR', 'USD', 'GBP'], which means,
          GBP -> EUR -> USD -> GBP, where it ALWAYS starts AND returns to GBP
    '''
    Amount = investment # Starts at initial investment
    for i in range(len(path)-1):
        from_currency = currencies.index(path[i])  # Takes the i'th element of path (the VERY first will always be GBP) and finds and finds the index in relation to currencies
        to_currency = currencies.index(path[i+1]) # Finds the currency as an index next in the path, the one that is being exchanged to.
        Amount *= Exchange_Matrix[from_currency, to_currency] # Finds the exchange rate and multiplies them all together, along with investment
    return Amount

def generate_paths(max_depth: int=3) -> dict:
    '''
    Generates all possible paths using permutations. USD, EUR, and CAD are all shuffled,
    where one exchange is allowed, two are, then three. GBP is "added" on to the left and right
    of the path at the end since the start and end point.
    '''
    results = {} # Creates dictionary
    foreign_exchange = currencies[1:] # Takes all except the first element (GBP)

    for depth in range(1, max_depth +1):
        for path in permutations(foreign_exchange, depth):
            full_path = ['GBP'] + list(path) + ['GBP']
            final_gbp = exchange_gbp(full_path) # Plugs path into exchange_gbp function
            results[tuple(full_path)] = final_gbp - investment # Sets up tuple of path:profit

    return results # Dictionary: {path: profit}

def answer(max_depth: int=3) -> tuple:
    '''
    This uses the generate_paths function to find the one that produces the maximum profit
    '''
    results = generate_paths(max_depth)
    best_path = max(results, key=results.get)
    return best_path, results[best_path]

best_path , best_profit = answer(3)
# results = generate_paths(3)
# print(results)
print('Opportunity 1 --------------------------------')
print()
print(f'Best exchange: {best_path}')
print(f'Profit: {best_profit:.2f} GBP')
print()
# Opportunity 2

# Key info: data available 01/02/2025. Future that delivers on 01/09/2025.
# Prices are given in GBX (pence)

Future_Dividends = np.matrix([
                              [439,61,327],
                              [2714,220,2424],
                              [12395,410,11879],
                              [8862,256,8569]
                              ])

def Fair_price_x(matrix):
    results = {}
    dividend_interest_inv = np.exp(-r * (2/12)) + np.exp(-r * (5/12))
    
    for i in range(matrix.shape[0]):
        FairPrice = round((matrix[i,0] - (matrix[i,1] * (dividend_interest_inv))) * np.exp(r * (7/12)),2)
        X = matrix[i,2]
        results[X] = FairPrice
    
    for X, FairPrice in results.items():
        
        print(f'Strike Price: {X}:, Fair Price: {FairPrice}')
        
    return results

def Arbritrage_2(matrix):
    results_XGREATER = {}
    results_S0GREATER = {}
    results = Fair_price_x(matrix)
    dividend_interest_T = np.exp(r * ((7/12)-(2/12))) + np.exp(r * ((7/12)-(5/12)))
    for X, FairPrice in results.items():
        if X > FairPrice:
            row = np.where(matrix[:, 2] == X)[0][0] # Finds the index of X in the given matrix
            initial_investment = matrix[row,0] # Buy the asset 
            profit = (matrix[row,2]) + (matrix[row,1] * dividend_interest_T) - (initial_investment) # Profit is X + D_T - S_0
            no_assets = investment//(matrix[row,0]/100) # Calculating number of assets, prices are given in pence, thus dividing prices by 100 give pounds.
            change = investment/(matrix[row,0]/100) -  no_assets # Number of assets obtained - fraction of asset left over
            correct_change = change * (matrix[row,0]/100) # Mulitplying this fraction by S_0 gives the amount that was left over when buying the number of assets - this will be put in bonds to gain interest overtime
            profit_converted = ((profit/100) * (no_assets)) + (correct_change * np.exp(r * (7/12))) - correct_change # Adding all together and converting to pounds - attaching the interst in bonds and removing the amount put into bonds, minus change since the interest it gains over time is the profit
            results_XGREATER[X] = round(profit_converted,2) # Putting in dictionary
        else:
            '''
            row2 = np.where(matrix[:, 2] == X)[0][0]
            initial_investment2 = matrix[row2,0] # Buy bonds worth S_0 with interest rate r
            profit2 = (initial_investment2 * np.exp(r* (7/12))) # S_0 gains interest r over T
            no_assets2 = investment//(matrix[row2,0]/100)
            change2 = investment/(matrix[row2,0]/100) - no_assets2
            correct_change2 = change2 * (matrix[row2,0]/100)
            '''
            # Buy bonds for 10,000 with interest r
            profit_converted2 = (investment * np.exp(r*(7/12))) - investment
            results_S0GREATER[X] = round(profit_converted2,2)
    
    print('X > (S_0 - D_0)e^rT')
    for X, profit in results_XGREATER.items():
        print(f'Strike Price: {X}:, Arbritrage profit: {profit}')

    print('X < (S_0 - D_0)e^rT')
    for X, profit in results_S0GREATER.items():
        print(f'Strike Price: {X}:, Arbritrage profit: {profit}')
    
    return results_XGREATER, results_S0GREATER

def bestArb2():
    results_XGREATER, results_S0GREATER = Arbritrage_2(Future_Dividends)
    results = {**results_XGREATER, **results_S0GREATER}
    x = max(results, key=results.get)
    print(f'X: {x}, Arb profit: {results[x]}')
    return x

print('Opportunity 2 --------------------------------')
print()
bestArb2() # To see everything remove comments in opportunity 2's functions
print()

# Opportunity 3

# Data available 1st of Feb. Future strike price X is given on the 1st Oct.
# All prices are GBP per ounce, storage costs are given per ounce
# Payed per four month period
Cost_of_carry_matrix = np.matrix([
                                  [2927.22,122.94,3137.06],
                                  [33.18,2.05,38.20],
                                  [1047.21,51.22,1129.62],
                                  [4.55,0.31,5.00]])

def Fair_price_x_UPFRONT(matrix):
    # Paying immediately for storage.
    results = {}
    
    for i in range(matrix.shape[0]):
        FairPrice = round((matrix[i,0] + (matrix[i,1] * 2)) * np.exp(r * (8/12)),2) #X = (S_0 + U_0)e^rT
        X = matrix[i,2]
        results[X] = FairPrice
    
    for X, FairPrice in results.items():
        
        print(f'Strike Price: {X}:, Fair Price: {FairPrice}')
        
    return results

def Fair_price_x_overtime(matrix):
    # Paying immediately for storage.
    results = {}
    
    for i in range(matrix.shape[0]):
        FairPrice = round(((matrix[i,0] + (matrix[i,1] + (matrix[i,1]*np.exp(-r*(4/12))))) * np.exp(r * (8/12))),2) #X = (S_0 + U_0)e^rT
        X = matrix[i,2]
        results[X] = FairPrice
    
    for X, FairPrice in results.items():
        
        print(f'Strike Price: {X}:, Fair Price: {FairPrice}')
        
    return results

# def Arbritrage_3(matrix):
#     results_XGREATER = {}
#     results_S0GREATER = {}
#     results = Fair_price_x_UPFRONT(matrix)
#     for X, FairPrice in results.items():
#         if X > FairPrice:
#             row = np.where(matrix[:, 2] == X)[0][0] # Finds the index of X in the given matrix
#             no_assets = investment//(matrix[row,0] + matrix[row,1])
#             change = matrix[row,0] * (investment/(matrix[row,0] + matrix[row,1]) - no_assets)
#             profit = matrix[row,2] - (matrix[row,1] *2) - matrix[row,0] # X - U_0 - S_0 
#             final_profit = (profit * no_assets) + (change * np.exp(r*(8/12))) - change
#             results_XGREATER[X] = round(final_profit,2) # Putting in dictionary
#         else:
#             '''
#             row2 = np.where(matrix[:, 2] == X)[0][0]
#             initial_investment2 = matrix[row2,0] # Buy bonds worth S_0 with interest rate r
#             profit2 = (initial_investment2 * np.exp(r* (7/12))) # S_0 gains interest r over T
#             no_assets2 = investment//(matrix[row2,0]/100)
#             change2 = investment/(matrix[row2,0]/100) - no_assets2
#             correct_change2 = change2 * (matrix[row2,0]/100)
#             '''
#             # Buy bonds for 10,000 with interest r
#             profit_converted3 = (investment * np.exp(r*(8/12))) - investment
#             results_S0GREATER[X] = round(profit_converted3,2)

print('Opportunity 3 --------------------------------')
print()
print('Paying for storage immediately')
Fair_price_x_UPFRONT(Cost_of_carry_matrix)
print()
print('Paying for storage per 4 months')
Fair_price_x_overtime(Cost_of_carry_matrix)

# Opp 4:

Option_trading_matrix = np.matrix([
                                   [12128, 69813, 70174, 12806],
                                   [151570,83222,91507,163915],
                                   [12626, 7137, 7998, 13830],
                                   [453,29,75,512]])

def PutCallParity(matrix):
    results = {}
    # C - P = S_0 - Ke^-rT
    # C = call premium
    # P = put premium
    diff = []
    for i in range(matrix.shape[0]):

        lhs = round((matrix[i,1] - matrix[i,2]),2) # C - P
        rhs = round((matrix[i,0] - (matrix[i,3]*np.exp(-r*(7/12)))),2) # S_0 - Ke^-rT
        results[lhs] = rhs
        diff.append(lhs - rhs)

    for lhs, rhs in results.items():
        
        print(f'Strike Price: {lhs}:, Fair Price: {rhs}')
    print()
    print('Difference in pounds')
    for i in diff:
        print(i/100)
    
    return results

print()
print('Opportunity 4 --------------------------------')
print()
print('In order of row 1 to row 4')
PutCallParity(Option_trading_matrix)