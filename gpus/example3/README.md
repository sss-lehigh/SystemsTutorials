# Example 3: Group By

Group by is a relational algebra operation used by databases for
proccessing and agregating data.

For example in this we consider a coffee buisness which has locations
in multiple cities. This database stores records of the location and
how much money the location made.

We assume the table looks like the following:

| CityID | LocationID | Profit |
|--------|------------|--------|
| 0      | 0          | $100   |
| 0      | 1          | $200   |

And we have projected the data to just get city and profit:

| CityID | Profit |
|--------|--------|
| 0      | $100   |
| 0      | $200   |

At this point we can group by the cityID and agregate the profit.

For this example it would yield that city 0 has a profit of $300.

Processing these kinds of operations can be done in parallel on both
the CPU and GPU in this example we look at ways that this can be done
and current limitations of GPU programming (not all algorithms are easily
portable to the GPU).

