#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <chrono>

using namespace std;

// ============================================================
// Constants
// ============================================================

const double SEQ_RMAX = 0.016;
const double SEQ_EPS = 1e-9;

const int SEQ_MAX_WEIGHT_UNITS = 4; // 20% = 4 units
const int SEQ_TOTAL_UNITS = 20;     // 100% = 20 units

// ============================================================
// Result structure
// ============================================================

struct SequentialResult
{
    vector<string> tickers;

    vector<double> bestAllocation;

    long long evaluatedPortfolios = 0;
    long long riskFeasiblePortfolios = 0;

    double bestReturn =
        -numeric_limits<double>::infinity();

    double bestRisk = 0.0;

    double executionTime = 0.0;
};

// ============================================================
// Load optimization input
//
// Reads:
// data/Optimization_Input-Table 1.csv
//
// Columns used:
// Ticker
// Expected Return (Daily)
// ============================================================

bool loadSequentialExpectedReturns(
    const string &filename,
    vector<string> &tickers,
    vector<double> &expectedReturns)
{
    ifstream file(filename);

    if (!file.is_open())
    {
        cerr << "Sequential search error: Could not open "
             << filename << endl;

        return false;
    }

    string line;

    // Skip header
    getline(file, line);

    while (getline(file, line))
    {
        if (line.empty())
        {
            continue;
        }

        stringstream ss(line);

        string ticker;
        string returnValue;

        getline(ss, ticker, ',');
        getline(ss, returnValue, ',');

        if (ticker.empty() || returnValue.empty())
        {
            continue;
        }

        tickers.push_back(ticker);
        expectedReturns.push_back(
            stod(returnValue));
    }

    file.close();

    return true;
}

// ============================================================
// Load covariance matrix
//
// Reads:
// data/Covariance-Table 1.csv
//
// First column = ticker
// Remaining 20 columns = covariance values
// ============================================================

bool loadSequentialCovariance(
    const string &filename,
    vector<vector<double>> &covariance)
{
    ifstream file(filename);

    if (!file.is_open())
    {
        cerr << "Sequential search error: Could not open "
             << filename << endl;

        return false;
    }

    string line;

    // Skip header
    getline(file, line);

    while (getline(file, line))
    {
        if (line.empty())
        {
            continue;
        }

        stringstream ss(line);

        string value;

        // Skip ticker
        getline(ss, value, ',');

        vector<double> row;

        while (getline(ss, value, ','))
        {
            if (!value.empty())
            {
                row.push_back(stod(value));
            }
        }

        if (!row.empty())
        {
            covariance.push_back(row);
        }
    }

    file.close();

    return true;
}

// ============================================================
// Calculate portfolio expected return
//
// Return = Σ(w_i × μ_i)
// ============================================================

double calculateSequentialReturn(
    const vector<double> &allocation,
    const vector<double> &expectedReturns)
{
    double result = 0.0;

    for (int i = 0;
         i < static_cast<int>(allocation.size());
         ++i)
    {
        result +=
            allocation[i] *
            expectedReturns[i];
    }

    return result;
}

// ============================================================
// Calculate portfolio risk
//
// Risk = sqrt(w^T Σ w)
// ============================================================

double calculateSequentialRisk(
    const vector<double> &allocation,
    const vector<vector<double>> &covariance)
{
    int n =
        static_cast<int>(allocation.size());

    double variance = 0.0;

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            variance +=
                allocation[i] *
                allocation[j] *
                covariance[i][j];
        }
    }

    if (variance < 0.0 &&
        variance > -SEQ_EPS)
    {
        variance = 0.0;
    }

    return sqrt(max(0.0, variance));
}

// ============================================================
// Check allocation = 100%
// ============================================================

bool isSequentialValidAllocation(
    const vector<double> &allocation)
{
    double total = 0.0;

    for (double weight : allocation)
    {
        total += weight;
    }

    return abs(total - 1.0) < SEQ_EPS;
}

// ============================================================
// Sequential exhaustive recursion
//
// 1 unit = 5%
// 20 units = 100%
//
// Only mathematical feasibility pruning is used.
// No return-based or risk-based pruning is used.
//
// Therefore every feasible complete portfolio is evaluated.
// ============================================================

void sequentialSearch(
    int assetIndex,
    int remainingUnits,
    vector<double> &allocation,
    const vector<double> &expectedReturns,
    const vector<vector<double>> &covariance,
    long long &evaluatedPortfolios,
    long long &riskFeasiblePortfolios,
    double &bestReturn,
    double &bestRisk,
    vector<double> &bestAllocation)
{
    int n =
        static_cast<int>(allocation.size());

    // --------------------------------------------------------
    // Feasibility pruning
    // --------------------------------------------------------

    int remainingAssets =
        n - assetIndex;

    if (remainingUnits < 0)
    {
        return;
    }

    if (remainingUnits >
        remainingAssets * SEQ_MAX_WEIGHT_UNITS)
    {
        return;
    }

    // --------------------------------------------------------
    // Base case
    // --------------------------------------------------------

    if (assetIndex == n)
    {
        if (remainingUnits != 0)
        {
            return;
        }

        // This is a complete feasible allocation
        ++evaluatedPortfolios;
        if (evaluatedPortfolios % 100000 == 0)
        {
            cout << "Checked: "
                 << evaluatedPortfolios
                 << endl;
        }

        // Extra verification using actual percentages
        if (!isSequentialValidAllocation(allocation))
        {
            return;
        }

        // ----------------------------------------------------
        // Calculate risk
        // ----------------------------------------------------

        double currentRisk =
            calculateSequentialRisk(
                allocation,
                covariance);

        // ----------------------------------------------------
        // Risk constraint
        // ----------------------------------------------------

        if (currentRisk >
            SEQ_RMAX + SEQ_EPS)
        {
            return;
        }

        ++riskFeasiblePortfolios;

        // ----------------------------------------------------
        // Calculate expected return
        // ----------------------------------------------------

        double currentReturn =
            calculateSequentialReturn(
                allocation,
                expectedReturns);

        // ----------------------------------------------------
        // Update best portfolio
        // ----------------------------------------------------

        if (currentReturn > bestReturn)
        {
            bestReturn = currentReturn;

            bestRisk = currentRisk;

            bestAllocation = allocation;
        }

        return;
    }

    // --------------------------------------------------------
    // Find valid weight range
    // --------------------------------------------------------

    int assetsAfter =
        n - assetIndex - 1;

    int minUnits =
        max(
            0,
            remainingUnits -
                SEQ_MAX_WEIGHT_UNITS * assetsAfter);

    int maxUnits =
        min(
            SEQ_MAX_WEIGHT_UNITS,
            remainingUnits);

    // --------------------------------------------------------
    // Try every allowed weight
    //
    // 0%, 5%, 10%, 15%, 20%
    // --------------------------------------------------------

    for (int units = minUnits;
         units <= maxUnits;
         ++units)
    {
        allocation[assetIndex] =
            units * 0.05;

        sequentialSearch(
            assetIndex + 1,
            remainingUnits - units,
            allocation,
            expectedReturns,
            covariance,
            evaluatedPortfolios,
            riskFeasiblePortfolios,
            bestReturn,
            bestRisk,
            bestAllocation);
    }

    // Backtrack
    allocation[assetIndex] = 0.0;
}

// ============================================================
// FUNCTION CALLED FROM main.cpp
// ============================================================

void runSequentialExhaustive()
{
    cout << "\n";
    cout << "============================================"
         << endl;
    cout << "     SEQUENTIAL EXHAUSTIVE SEARCH"
         << endl;
    cout << "============================================"
         << endl;

    // --------------------------------------------------------
    // File paths
    // --------------------------------------------------------

    const string expectedReturnFile =
        "data/Optimization_Input-Table 1.csv";

    const string covarianceFile =
        "data/Covariance-Table 1.csv";

    // --------------------------------------------------------
    // Load data
    // --------------------------------------------------------

    vector<string> tickers;
    vector<double> expectedReturns;
    vector<vector<double>> covariance;

    if (!loadSequentialExpectedReturns(
            expectedReturnFile,
            tickers,
            expectedReturns))
    {
        return;
    }

    if (!loadSequentialCovariance(
            covarianceFile,
            covariance))
    {
        return;
    }

    // --------------------------------------------------------
    // Validate data
    // --------------------------------------------------------

    int n =
        static_cast<int>(tickers.size());

    if (n == 0)
    {
        cout << "No assets loaded." << endl;
        return;
    }

    if (expectedReturns.size() !=
        static_cast<size_t>(n))
    {
        cout << "Error: Expected return count "
             << "does not match asset count."
             << endl;

        return;
    }

    if (covariance.size() !=
        static_cast<size_t>(n))
    {
        cout << "Error: Covariance matrix row count "
             << "does not match asset count."
             << endl;

        return;
    }

    for (int i = 0; i < n; ++i)
    {
        if (covariance[i].size() !=
            static_cast<size_t>(n))
        {
            cout << "Error: Covariance matrix is not "
                 << "square."
                 << endl;

            return;
        }
    }

    cout << "Number of assets: "
         << n
         << endl;

    cout << "Risk Limit (Rmax): "
         << fixed
         << setprecision(10)
         << SEQ_RMAX
         << endl;

    // --------------------------------------------------------
    // Initialize search
    // --------------------------------------------------------

    vector<double> allocation(
        n,
        0.0);

    vector<double> bestAllocation(
        n,
        0.0);

    long long evaluatedPortfolios = 0;

    long long riskFeasiblePortfolios = 0;

    double bestReturn =
        -numeric_limits<double>::infinity();

    double bestRisk = 0.0;

    // --------------------------------------------------------
    // Start timer
    // --------------------------------------------------------

    auto startTime =
        chrono::high_resolution_clock::now();

    // --------------------------------------------------------
    // Run sequential exhaustive search
    // --------------------------------------------------------

    sequentialSearch(
        0,
        SEQ_TOTAL_UNITS,
        allocation,
        expectedReturns,
        covariance,
        evaluatedPortfolios,
        riskFeasiblePortfolios,
        bestReturn,
        bestRisk,
        bestAllocation);

    // --------------------------------------------------------
    // Stop timer
    // --------------------------------------------------------

    auto endTime =
        chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed =
        endTime - startTime;

    // --------------------------------------------------------
    // Display results
    // --------------------------------------------------------

    cout << "\n"
         << "Evaluated Portfolios: "
         << evaluatedPortfolios
         << endl;

    cout << "Risk-Feasible Portfolios: "
         << riskFeasiblePortfolios
         << endl;

    if (bestReturn ==
        -numeric_limits<double>::infinity())
    {
        cout << "\nNo portfolio satisfies Rmax."
             << endl;
    }
    else
    {
        cout << "\nOptimal Portfolio:"
             << endl;

        cout << "\nExpected Return (daily): "
             << bestReturn
             << endl;

        cout << "Portfolio Risk (daily):  "
             << bestRisk
             << endl;

        cout << "Rmax:                    "
             << SEQ_RMAX
             << endl;

        cout << "\nPortfolio Allocation:"
             << endl;

        cout << left
             << setw(10)
             << "Ticker"
             << setw(15)
             << "Allocation"
             << endl;

        cout << "-----------------------------------"
             << endl;

        double totalAllocation = 0.0;

        for (int i = 0; i < n; ++i)
        {
            if (bestAllocation[i] > SEQ_EPS)
            {
                cout << left
                     << setw(10)
                     << tickers[i]
                     << setw(15)
                     << fixed
                     << setprecision(0)
                     << bestAllocation[i] * 100.0
                     << "%"
                     << endl;
            }

            totalAllocation +=
                bestAllocation[i];
        }

        cout << "\nTotal Allocation: "
             << fixed
             << setprecision(0)
             << totalAllocation * 100.0
             << "%"
             << endl;
    }

    cout << "\nSequential Execution Time: "
         << fixed
         << setprecision(6)
         << elapsed.count()
         << " seconds"
         << endl;

    cout << "============================================"
         << endl;
}
