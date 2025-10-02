show tables;


create or replace table review_competitors
(
	id Int32,
	competitor_banks String, 
	has_competitor_banks Bool,
)
engine MergeTree
primary key ();


create or replace table review_full
engine MergeTree
primary key (product, dateCreate)
as
select t.id as id, * except('id$')
	from review t
	left join (
		select id, groupArray(class) as classes
			from review_classes
			group by all
	) tt on t.id = tt.id
	left join (
		select distinct toUInt64(id) as id, splitByChar(',', competitor_banks) as competitor_banks, has_competitor_banks
			from review_competitors
	) ttt on t.id = ttt.id;

select * from review_full;
