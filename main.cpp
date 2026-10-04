#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <iomanip>
#include <omp.h>
#include <cstdlib>

using namespace std;

// ============================================================
// Sequential Exhaustive Search
// Implemented in src/sequential_exhaustive.cpp
// ============================================================

void runSequentialExhaustive();

/*
    Exact Exhaustive Portfolio Optimization with OpenMP

    Data:
      - 20 stocks from the final preprocessing file
      - Daily expected returns
      - Daily covariance matrix

    Portfolio:
      - Each stock weight is 0%, 5%, 10%, 15%, or 20%
      - Total allocation = 100%
      - Risk <= Rmax
      - Objective: maximize expected return

    IMPORTANT:
      This is an exhaustive method.
      No branch-and-bound, heuristic pruning, greedy selection,
      random search, or approximation is used.

    OpenMP parallelizes the first two weight decisions:
      5 x 5 = 25 independent search branches.

    A branch is allowed to choose only weights that can still
    complete the required total allocation. This does NOT remove
    feasible portfolios; it simply avoids generating partial
    assignments that can never form a valid 100% portfolio.

    Usage:
      ./portfolio_optimizer
      ./portfolio_optimizer 0.016

    Rmax is a DAILY portfolio-risk limit.
*/

struct Solution
{
    double returnValue;
    double risk;
    vector<int> weights;

    Solution()
        : returnValue(-numeric_limits<double>::infinity()),
          risk(0.0) {}
};

double calculateReturn(
    const vector<int> &weights,
    const vector<double> &expectedReturn)
{
    double portfolioReturn = 0.0;

    for (int i = 0; i < static_cast<int>(weights.size()); ++i)
    {
        double w = weights[i] / 20.0;
        portfolioReturn += w * expectedReturn[i];
    }

    return portfolioReturn;
}

double calculateRisk(
    const vector<int> &weights,
    const vector<vector<double>> &covariance)
{
    const int n = static_cast<int>(weights.size());

    double variance = 0.0;

    for (int i = 0; i < n; ++i)
    {
        double wi = weights[i] / 20.0;

        for (int j = 0; j < n; ++j)
        {
            double wj = weights[j] / 20.0;
            variance += wi * wj * covariance[i][j];
        }
    }

    // Guard against tiny negative floating-point roundoff.
    variance = max(0.0, variance);

    return sqrt(variance);
}

/*
    Exhaustive recursive search.

    remainingWeight is measured in 5% units:
      1 unit = 5%
      20 units = 100%

    partialReturn is the return contribution accumulated so far.

    partialVariance is the exact variance contribution of the
    currently assigned stocks:
        sum_{i,j in assigned} wi * wj * covariance[i][j]

    Updating the partial variance incrementally avoids repeatedly
    calculating the full 20 x 20 quadratic form at every leaf.
*/
void search(
    int index,
    int N,
    int remainingWeight,
    vector<int> &currentWeights,
    const vector<double> &expectedReturn,
    const vector<vector<double>> &covariance,
    double Rmax,
    double partialReturn,
    double partialVariance,
    Solution &localBest,
    unsigned long long &evaluatedPortfolios)
{
    // No valid completion is possible if more than 100% was assigned.
    if (remainingWeight < 0)
    {
        return;
    }

    // Last stock: its weight is forced by the remaining allocation.
    if (index == N - 1)
    {

        if (remainingWeight < 0 || remainingWeight > 4)
        {
            return;
        }

        currentWeights[index] = remainingWeight;

        const double w = remainingWeight / 20.0;

        // Add the last stock's exact contribution to variance.
        double variance = partialVariance;

        for (int j = 0; j < index; ++j)
        {
            const double wj = currentWeights[j] / 20.0;

            // Two symmetric covariance terms:
            // wi*wj*Cov(i,j) + wj*wi*Cov(j,i)
            variance +=
                2.0 * w * wj * covariance[index][j];
        }

        variance +=
            w * w * covariance[index][index];

        const double risk = sqrt(max(0.0, variance));

        // This is one complete feasible portfolio.
        ++evaluatedPortfolios;

        if (risk <= Rmax)
        {

            const double portfolioReturn =
                partialReturn + w * expectedReturn[index];

            if (portfolioReturn > localBest.returnValue)
            {

                localBest.returnValue = portfolioReturn;
                localBest.risk = risk;
                localBest.weights = currentWeights;
            }
        }

        return;
    }

    /*
        Choose the current stock's weight.

        We do not generate values that make it impossible to
        distribute the remaining weight among the remaining stocks.

        This is exact enumeration of all feasible portfolios,
        not heuristic pruning.
    */

    const int stocksAfter = N - index - 1;

    const int minWeight = max(
        0,
        remainingWeight - 4 * stocksAfter);

    const int maxWeight = min(
        4,
        remainingWeight);

    for (int weight = minWeight;
         weight <= maxWeight;
         ++weight)
    {

        currentWeights[index] = weight;

        const double w = weight / 20.0;

        // Incremental return.
        const double newReturn =
            partialReturn +
            w * expectedReturn[index];

        // Incremental variance:
        // previous variance
        // + 2*w*sum(previous_wj*Cov(index,j))
        // + w^2*Cov(index,index)
        double newVariance = partialVariance;

        for (int j = 0; j < index; ++j)
        {
            const double wj = currentWeights[j] / 20.0;

            newVariance +=
                2.0 * w * wj * covariance[index][j];
        }

        newVariance +=
            w * w * covariance[index][index];

        search(
            index + 1,
            N,
            remainingWeight - weight,
            currentWeights,
            expectedReturn,
            covariance,
            Rmax,
            newReturn,
            newVariance,
            localBest,
            evaluatedPortfolios);
    }
}

Solution parallelExhaustiveSearch(
    int N,
    const vector<double> &expectedReturn,
    const vector<vector<double>> &covariance,
    double Rmax,
    unsigned long long &totalEvaluated)
{
    Solution globalBest;

    totalEvaluated = 0;

    if (N <= 0)
    {
        return globalBest;
    }

    /*
        There are 25 first-two-stock combinations:
            Stock 1 = 0..4 units
            Stock 2 = 0..4 units

        Only combinations that can be completed to 20 units
        are submitted to the recursive exhaustive search.
    */

#pragma omp parallel
    {
        Solution localBest;
        unsigned long long localEvaluated = 0;

        vector<int> currentWeights(N, 0);

#pragma omp for schedule(static)
        for (int branch = 0; branch < 25; ++branch)
        {

            const int firstWeight = branch / 5;
            const int secondWeight = branch % 5;

            const int assigned =
                firstWeight + secondWeight;

            if (assigned > 20)
            {
                continue;
            }

            const int remainingWeight =
                20 - assigned;

            // With N=20, 18 stocks remain after the first two.
            // This branch is always completable for the current
            // 0..4-unit stock bounds, but keep the check general.
            if (remainingWeight > 4 * (N - 2))
            {
                continue;
            }

            currentWeights[0] = firstWeight;
            currentWeights[1] = secondWeight;

            const double w0 = firstWeight / 20.0;
            const double w1 = secondWeight / 20.0;

            double partialReturn =
                w0 * expectedReturn[0] +
                w1 * expectedReturn[1];

            double partialVariance =
                w0 * w0 * covariance[0][0] +
                w1 * w1 * covariance[1][1] +
                2.0 * w0 * w1 * covariance[0][1];

            search(
                2,
                N,
                remainingWeight,
                currentWeights,
                expectedReturn,
                covariance,
                Rmax,
                partialReturn,
                partialVariance,
                localBest,
                localEvaluated);
        }

#pragma omp atomic
        totalEvaluated += localEvaluated;

#pragma omp critical
        {
            if (localBest.returnValue >
                globalBest.returnValue)
            {

                globalBest = localBest;
            }
        }
    }

    return globalBest;
}

int main(int argc, char *argv[])
{

    constexpr int MAX_ASSETS = 20;

    int N = 20;
    int requestedThreads = 0;
    double Rmax = 0.016;

    if (argc >= 2)
        N = atoi(argv[1]);
    if (argc >= 3)
        requestedThreads = atoi(argv[2]);
    if (argc >= 4)
        Rmax = atof(argv[3]);

    if (N < 2 || N > MAX_ASSETS)
    {
        cerr << "Error: number of assets must be between 2 and 20.\n";
        return 1;
    }

    if (requestedThreads > 0)
    {
        omp_set_num_threads(requestedThreads);
    }

    const vector<string> tickers = {

        "AZO",
        "AME",
        "ACGL",
        "AAPL",
        "DECK",
        "EME",
        "APH",
        "BLK",
        "BRO",
        "ATO",
        "ETN",
        "DHR",
        "FICO",
        "EW",
        "FAST",
        "COST",
        "BALL",
        "AJG",
        "DE",
        "DVA"};

    const vector<double> allExpectedReturn = {

        0.000913531714854,
        0.0008588494811,
        0.00076506113199,
        0.00116295803295,
        0.00151067110788,
        0.00104555550055,
        0.000946556631548,
        0.000898639255529,
        0.000683947228577,
        0.000598452089864,
        0.00078177344564,
        0.000745705478906,
        0.00100899311792,
        0.000849683429001,
        0.00081190797965,
        0.000647116236716,
        0.000726548393291,
        0.000654363350499,
        0.000834128930788,
        0.00102420438732};

    const vector<vector<double>> allCovariance = {

        {0.000295680305036, 0.000108081477088, 8.11602162936e-05, 0.000103079171623, 0.000126948048718, 0.000127763004348, 0.000113073532805, 0.000117620516567, 8.61085538842e-05, 6.75099121659e-05, 0.000113480881399, 9.26458284759e-05, 0.000114470580798, 6.99850923032e-05, 0.000129409686003, 0.000112284219865, 9.43400653108e-05, 7.99720076466e-05, 0.000116097578887, 7.89918509114e-05},
        {0.000108081477088, 0.000297074379081, 0.000108688447764, 0.000151805018142, 0.000184044799707, 0.000202273844883, 0.000193012867177, 0.000182757685601, 0.000121455453082, 9.75259196728e-05, 0.000200504767398, 0.000142402263927, 0.000178946446235, 0.000115533539385, 0.000176770630316, 0.000105624761043, 0.000135898756365, 0.000110022240032, 0.000186417506273, 0.000101971545529},
        {8.11602162936e-05, 0.000108688447764, 0.000276450744587, 9.02123003172e-05, 0.000122954571521, 0.000133168840588, 0.000101694200347, 0.000132051305904, 0.000108632740885, 7.60941668188e-05, 0.00011843150268, 7.86588227416e-05, 0.000124580775279, 7.47483646939e-05, 0.000100777917989, 7.05149708309e-05, 8.92785465514e-05, 9.79930918534e-05, 0.00011172740159, 7.1575304614e-05},
        {0.000103079171623, 0.000151805018142, 9.02123003172e-05, 0.000604586318634, 0.000188708510271, 0.000158619012191, 0.000193932195912, 0.000178766273531, 0.000100458732209, 8.24161801596e-05, 0.000166675754858, 0.000146424645275, 0.000184763629764, 0.000113689035742, 0.000170466550266, 0.00013888475832, 0.000124548554589, 8.81715209061e-05, 0.000169873054818, 8.01030030839e-05},
        {0.000126948048718, 0.000184044799707, 0.000122954571521, 0.000188708510271, 0.00109564715266, 0.000233347524022, 0.000206257941642, 0.000207386780755, 0.000117133864302, 6.90681688673e-05, 0.000207530760516, 0.000153353851144, 0.000207316210482, 0.000124462067563, 0.00019419326786, 0.00011244151889, 0.000141601174882, 0.000115845719588, 0.000187305735675, 8.59135504779e-05},
        {0.000127763004348, 0.000202273844883, 0.000133168840588, 0.000158619012191, 0.000233347524022, 0.000512190617059, 0.000236421587607, 0.000216965173289, 0.000135118585129, 9.86218629038e-05, 0.000244727417571, 0.000152043328177, 0.000212375463047, 0.000109012006549, 0.00019718147279, 0.000123893267649, 0.000149664443016, 0.000114995432876, 0.000222020520639, 0.000122670487879},
        {0.000113073532805, 0.000193012867177, 0.000101694200347, 0.000193932195912, 0.000206257941642, 0.000236421587607, 0.000455320076402, 0.00020831555577, 0.000116808379398, 8.63888079672e-05, 0.000218004270971, 0.000163310014741, 0.000194569313412, 0.000122554885319, 0.000188441974213, 0.000129077471239, 0.000134045993987, 0.000100427261537, 0.000185206322832, 0.000108307402427},
        {0.000117620516567, 0.000182757685601, 0.000132051305904, 0.000178766273531, 0.000207386780755, 0.000216965173289, 0.00020831555577, 0.000442292089711, 0.000144025084032, 0.000100473155017, 0.000200455979845, 0.000155699240774, 0.000217483842154, 0.000128191149253, 0.000193669745507, 0.000122441405817, 0.000150512277075, 0.000133765584906, 0.000196919242389, 0.000109936535315},
        {8.61085538842e-05, 0.000121455453082, 0.000108632740885, 0.000100458732209, 0.000117133864302, 0.000135118585129, 0.000116808379398, 0.000144025084032, 0.000256539938857, 7.88359150909e-05, 0.000121018646081, 0.000103713596181, 0.000142432681962, 9.07105393067e-05, 0.000131521921609, 9.26861571518e-05, 0.000106324213285, 0.000147795690014, 0.000115504724779, 8.71043195465e-05},
        {6.75099121659e-05, 9.75259196728e-05, 7.60941668188e-05, 8.24161801596e-05, 6.90681688673e-05, 9.86218629038e-05, 8.63888079672e-05, 0.000100473155017, 7.88359150909e-05, 0.000186424000075, 9.45586766833e-05, 7.85030257058e-05, 0.000100840386719, 7.5720934368e-05, 9.27198334326e-05, 7.22088738858e-05, 9.37932993568e-05, 7.52522072959e-05, 8.47500480787e-05, 6.7983217083e-05},
        {0.000113480881399, 0.000200504767398, 0.00011843150268, 0.000166675754858, 0.000207530760516, 0.000244727417571, 0.000218004270971, 0.000200455979845, 0.000121018646081, 9.45586766833e-05, 0.000357706234396, 0.00016302613469, 0.000181787759893, 0.000111626576451, 0.000191042798565, 0.000118403818373, 0.000148781267056, 0.000112189922868, 0.000221070108248, 0.000105451896234},
        {9.26458284759e-05, 0.000142402263927, 7.86588227416e-05, 0.000146424645275, 0.000153353851144, 0.000152043328177, 0.000163310014741, 0.000155699240774, 0.000103713596181, 7.85030257058e-05, 0.00016302613469, 0.000314962361069, 0.000150179504115, 9.7603920711e-05, 0.000149601789159, 0.000110148050085, 0.000124456885582, 9.00232851342e-05, 0.000149257720734, 8.6806903748e-05},
        {0.000114470580798, 0.000178946446235, 0.000124580775279, 0.000184763629764, 0.000207316210482, 0.000212375463047, 0.000194569313412, 0.000217483842154, 0.000142432681962, 0.000100840386719, 0.000181787759893, 0.000150179504115, 0.00055917079824, 0.000130005648816, 0.000168336210253, 0.000117476748452, 0.000136650890994, 0.000124328543643, 0.000171326140269, 0.000109138401403},
        {6.99850923032e-05, 0.000115533539385, 7.47483646939e-05, 0.000113689035742, 0.000124462067563, 0.000109012006549, 0.000122554885319, 0.000128191149253, 9.07105393067e-05, 7.5720934368e-05, 0.000111626576451, 9.7603920711e-05, 0.000130005648816, 0.000437341303373, 0.000102038321258, 7.40959588661e-05, 9.56957241882e-05, 7.58104988232e-05, 9.53387895964e-05, 9.41753961979e-05},
        {0.000129409686003, 0.000176770630316, 0.000100777917989, 0.000170466550266, 0.00019419326786, 0.00019718147279, 0.000188441974213, 0.000193669745507, 0.000131521921609, 9.27198334326e-05, 0.000191042798565, 0.000149601789159, 0.000168336210253, 0.000102038321258, 0.000400994698748, 0.000137553074552, 0.000136332165603, 0.000120574397577, 0.000188497892967, 0.000101123810684},
        {0.000112284219865, 0.000105624761043, 7.05149708309e-05, 0.00013888475832, 0.00011244151889, 0.000123893267649, 0.000129077471239, 0.000122441405817, 9.26861571518e-05, 7.22088738858e-05, 0.000118403818373, 0.000110148050085, 0.000117476748452, 7.40959588661e-05, 0.000137553074552, 0.000272795184081, 9.75298611767e-05, 8.09898923011e-05, 0.000109756500591, 8.47238646584e-05},
        {9.43400653108e-05, 0.000135898756365, 8.92785465514e-05, 0.000124548554589, 0.000141601174882, 0.000149664443016, 0.000134045993987, 0.000150512277075, 0.000106324213285, 9.37932993568e-05, 0.000148781267056, 0.000124456885582, 0.000136650890994, 9.56957241882e-05, 0.000136332165603, 9.75298611767e-05, 0.000299122631638, 9.61967887767e-05, 0.000149911717285, 9.63626467319e-05},
        {7.99720076466e-05, 0.000110022240032, 9.79930918534e-05, 8.81715209061e-05, 0.000115845719588, 0.000114995432876, 0.000100427261537, 0.000133765584906, 0.000147795690014, 7.52522072959e-05, 0.000112189922868, 9.00232851342e-05, 0.000124328543643, 7.58104988232e-05, 0.000120574397577, 8.09898923011e-05, 9.61967887767e-05, 0.000257020794355, 0.000102275454794, 7.48253408554e-05},
        {0.000116097578887, 0.000186417506273, 0.00011172740159, 0.000169873054818, 0.000187305735675, 0.000222020520639, 0.000185206322832, 0.000196919242389, 0.000115504724779, 8.47500480787e-05, 0.000221070108248, 0.000149257720734, 0.000171326140269, 9.53387895964e-05, 0.000188497892967, 0.000109756500591, 0.000149911717285, 0.000102275454794, 0.000419504121124, 0.000105894122577},
        {7.89918509114e-05, 0.000101971545529, 7.1575304614e-05, 8.01030030839e-05, 8.59135504779e-05, 0.000122670487879, 0.000108307402427, 0.000109936535315, 8.71043195465e-05, 6.7983217083e-05, 0.000105451896234, 8.6806903748e-05, 0.000109138401403, 9.41753961979e-05, 0.000101123810684, 8.47238646584e-05, 9.63626467319e-05, 7.48253408554e-05, 0.000105894122577, 0.000477231602807}};

    // Use the first N assets for the asset-scaling experiment.
    vector<double> expectedReturn(
        allExpectedReturn.begin(),
        allExpectedReturn.begin() + N);

    vector<vector<double>> covariance(
        N, vector<double>(N));

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            covariance[i][j] = allCovariance[i][j];
        }
    }

    // The full source data contain 20 assets; for asset-scaling we
    // intentionally use the first N assets. Validate the selected data.
    if (static_cast<int>(expectedReturn.size()) != N ||
        static_cast<int>(covariance.size()) != N)
    {

        cerr << "Error: Selected input dimensions do not match N = "
             << N << ".\n";

        return 1;
    }

    for (const auto &row : covariance)
    {
        if (static_cast<int>(row.size()) != N)
        {
            cerr << "Error: Covariance matrix must be "
                    "20 x 20.\n";
            return 1;
        }
    }

    cout << fixed << setprecision(10);

    cout << "=============================================\n";
    cout << " Exact Exhaustive OpenMP Portfolio Optimizer\n";
    cout << "=============================================\n\n";

    cout << "Number of stocks: " << N << "\n";
    cout << "Weight increment: 5%\n";
    cout << "Total allocation: 100%\n";
    cout << "Risk limit (daily): " << Rmax << "\n";
    cout << "OpenMP threads: " << omp_get_max_threads() << "\n\n";

    /*
        For 20 stocks with each weight in {0,1,2,3,4}
        five-percent units and total = 20, the exact number
        of feasible portfolios is:

            coefficient of x^20 in (1+x+x^2+x^3+x^4)^20

        This equals 35,561,166,195.
    */
    // Exact count of feasible portfolios for the selected N.
    unsigned long long expectedPortfolios = 0;

    vector<unsigned long long> dp(21, 0);
    dp[0] = 1;

    for (int asset = 0; asset < N; ++asset)
    {
        vector<unsigned long long> next(21, 0);

        for (int total = 0; total <= 20; ++total)
        {
            for (int weight = 0; weight <= 4; ++weight)
            {
                if (total + weight <= 20)
                {
                    next[total + weight] += dp[total];
                }
            }
        }

        dp = next;
    }

    expectedPortfolios = dp[20];

    cout << "Expected feasible portfolios: "
         << expectedPortfolios << "\n\n";

    unsigned long long evaluatedPortfolios = 0;

    const double start = omp_get_wtime();

    Solution result =
        parallelExhaustiveSearch(
            N,
            expectedReturn,
            covariance,
            Rmax,
            evaluatedPortfolios);

    const double end = omp_get_wtime();

    cout << "Evaluated portfolios: "
         << evaluatedPortfolios << "\n";

    if (evaluatedPortfolios == expectedPortfolios)
    {
        cout << "Exhaustive check: PASS\n";
    }
    else
    {
        cout << "Exhaustive check: FAIL\n";
    }

    cout << "\nBest Portfolio:\n";

    if (result.weights.empty())
    {
        cout << "No feasible portfolio satisfies Rmax.\n";
    }
    else
    {

        double totalWeight = 0.0;

        for (int i = 0; i < N; ++i)
        {

            const double percentage =
                result.weights[i] * 5.0;

            totalWeight += percentage;

            cout << setw(6) << tickers[i]
                 << " : "
                 << setw(6) << percentage
                 << "%\n";
        }

        cout << "\nTotal Weight: "
             << totalWeight
             << "%\n";

        cout << "Best Expected Return (daily): "
             << result.returnValue
             << "\n";

        cout << "Portfolio Risk (daily): "
             << result.risk
             << "\n";

        cout << "Risk Limit (daily): "
             << Rmax
             << "\n";
    }

    cout << "\nExecution Time: "
         << end - start
         << " seconds\n";

    // ============================================================
    // Sequential Exhaustive Search
    // ============================================================

    runSequentialExhaustive();

    // ============================================================
    // END OF MY PART
    // ============================================================

    return 0;
}
