CREATE OR REPLACE FUNCTION `keepcoding.fnc_chk_clean_integer`(p_integer INT64) RETURNS INT64 AS (
( SELECT IFNULL(p_integer, -999999))
);