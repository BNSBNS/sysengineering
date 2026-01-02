"""Debug test for LIMIT parsing."""
import sqlglot
from sqlglot import exp
from db_engine.adapters.inbound import SQLParser, Limit

def test_limit_parse():
    # First, debug what sqlglot produces
    sql = 'SELECT id FROM users LIMIT 2'
    stmts = sqlglot.parse(sql, dialect='sqlite')
    stmt = stmts[0]
    limit = stmt.find(exp.Limit)
    print(f'Limit expression: {limit}')
    print(f'Limit.this: {limit.this}')
    print(f'Limit args: {limit.args}')
    print(f'Limit expressions: {limit.expressions}')

    # Check expression attribute
    if hasattr(limit, 'expression'):
        print(f'Limit.expression: {limit.expression}')

    # Check all attributes
    for key, val in limit.args.items():
        print(f'  limit.args[{key}]: {val} (type: {type(val).__name__})')
        if val and hasattr(val, 'this'):
            print(f'    .this: {val.this}')

    parser = SQLParser()
    plan = parser.parse(sql)
    print(f'Plan type: {type(plan).__name__}')
    if hasattr(plan, 'count'):
        print(f'Count: {plan.count}')

    assert isinstance(plan, Limit), f"Expected Limit but got {type(plan).__name__}"
    assert plan.count == 2, f"Expected count=2 but got {plan.count}"
