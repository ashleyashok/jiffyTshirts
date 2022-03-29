-- Part A: 1 How many orders were completed in 2018 ?
select
    count(*)
from
    (
        select
            (order_timestamp :: TIMESTAMP at time zone 'utc') at time zone 'est' as order_timestamp_est
        from
            orders
    ) a
where
    order_timestamp_est between '2018-01-01' and '2018-12-31'
    
 