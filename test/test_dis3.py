# Minimal tests for dis module

from __future__ import print_function

import difflib
import unittest
import sys
import io
import re
import types
import contextlib

import dis3


# Fix for printing to captured_output
class ForceUnicodeStringIO(io.StringIO):
    def write(self, s):
        s = unicode(s)
        return super(ForceUnicodeStringIO, self).write(s)


########################################################################
# origin: contextlib

class redirect_stdout:
    """Context manager for temporarily redirecting stdout to another file

        # How to send help() to stderr
        with redirect_stdout(sys.stderr):
            help(dir)

        # How to write help() to a file
        with open('help.txt', 'w') as f:
            with redirect_stdout(f):
                help(pow)
    """

    def __init__(self, new_target):
        self._new_target = new_target
        # We use a list of old targets to make this CM re-entrant
        self._old_targets = []

    def __enter__(self):
        self._old_targets.append(sys.stdout)
        sys.stdout = self._new_target
        return self._new_target

    def __exit__(self, exctype, excinst, exctb):
        sys.stdout = self._old_targets.pop()


########################################################################
# origin: test.support

def _filter_suite(suite, pred):
    """Recursively filter test cases in a suite based on a predicate."""
    newtests = []
    for test in suite._tests:
        if isinstance(test, unittest.TestSuite):
            _filter_suite(test, pred)
            newtests.append(test)
        else:
            if pred(test):
                newtests.append(test)
    suite._tests = newtests


def _run_suite(suite):
    """Run tests from a unittest.TestSuite-derived class."""
    if verbose:
        runner = unittest.TextTestRunner(sys.stdout, verbosity=2,
                                         failfast=failfast)
    else:
        runner = BasicTestRunner()

    result = runner.run(suite)
    if not result.wasSuccessful():
        if len(result.errors) == 1 and not result.failures:
            err = result.errors[0][1]
        elif len(result.failures) == 1 and not result.errors:
            err = result.failures[0][1]
        else:
            err = "multiple errors occurred"
            if not verbose: err += "; run in verbose mode for details"
        raise TestFailed(err)


def run_unittest(*classes):
    """Run tests from unittest.TestCase-derived classes."""
    valid_types = (unittest.TestSuite, unittest.TestCase)
    suite = unittest.TestSuite()
    for cls in classes:
        if isinstance(cls, str):
            if cls in sys.modules:
                suite.addTest(unittest.findTestCases(sys.modules[cls]))
            else:
                raise ValueError("str arguments must be keys in sys.modules")
        elif isinstance(cls, valid_types):
            suite.addTest(cls)
        else:
            suite.addTest(unittest.makeSuite(cls))
    def case_pred(test):
        if match_tests is None:
            return True
        for name in test.id().split("."):
            if fnmatch.fnmatchcase(name, match_tests):
                return True
        return False
    _filter_suite(suite, case_pred)
    _run_suite(suite)


@contextlib.contextmanager
def captured_output(stream_name):
    """Return a context manager used by captured_stdout/stdin/stderr
    that temporarily replaces the sys stream *stream_name* with a StringIO."""
    import io
    orig_stdout = getattr(sys, stream_name)
    setattr(sys, stream_name, ForceUnicodeStringIO())
    try:
        yield getattr(sys, stream_name)
    finally:
        setattr(sys, stream_name, orig_stdout)


def captured_stdout():
    """Capture the output of sys.stdout:

       with captured_stdout() as stdout:
           print("hello")
       self.assertEqual(stdout.getvalue(), "hello\n")
    """
    return captured_output("stdout")


########################################################################
# origin: test.bytecode_helper

_UNSPECIFIED = object()

class BytecodeTestCase(unittest.TestCase):
    """Custom assertion methods for inspecting bytecode."""

    def __init__(self, *args, **kwargs):
        self.maxDiff = None
        super(BytecodeTestCase, self).__init__(*args, **kwargs)

    def get_disassembly_as_string(self, co):
        s = io.StringIO()
        dis3.dis(co, file=s)
        return s.getvalue()

    def assertInBytecode(self, x, opname, argval=_UNSPECIFIED):
        """Returns instr if op is found, otherwise throws AssertionError"""
        for instr in dis3.get_instructions(x):
            if instr.opname == opname:
                if argval is _UNSPECIFIED or instr.argval == argval:
                    return instr
        disassembly = self.get_disassembly_as_string(x)
        if argval is _UNSPECIFIED:
            msg = '%s not found in bytecode:\n%s' % (opname, disassembly)
        else:
            msg = '(%s,%r) not found in bytecode:\n%s'
            msg = msg % (opname, argval, disassembly)
        self.fail(msg)

    def assertNotInBytecode(self, x, opname, argval=_UNSPECIFIED):
        """Throws AssertionError if op is found"""
        for instr in dis3.get_instructions(x):
            if instr.opname == opname:
                disassembly = self.get_disassembly_as_string(co)
                if opargval is _UNSPECIFIED:
                    msg = '%s occurs in bytecode:\n%s' % (opname, disassembly)
                elif instr.argval == argval:
                    msg = '(%s,%r) occurs in bytecode:\n%s'
                    msg = msg % (opname, argval, disassembly)
                self.fail(msg)


########################################################################
# origin: test.test_dis

def get_tb():
    def _error():
        try:
            1 / 0
        except Exception as e:
            tb = sys.exc_info()[2]
        return tb

    tb = _error()
    while tb.tb_next:
        tb = tb.tb_next
    return tb

TRACEBACK_CODE = get_tb().tb_frame.f_code

class _C:
    def __init__(self, x):
        self.x = x == 1

dis_c_instance_method = u"""\
%3d           0 LOAD_FAST                1 (x)
              3 LOAD_CONST               1 (1)
              6 COMPARE_OP               2 (==)
              9 LOAD_FAST                0 (self)
             12 STORE_ATTR               0 (x)
             15 LOAD_CONST               0 (None)
             18 RETURN_VALUE
""" % (_C.__init__.__code__.co_firstlineno + 1,)

dis_c_instance_method_bytes = u"""\
          0 LOAD_FAST                1 (1)
          3 LOAD_CONST               1 (1)
          6 COMPARE_OP               2 (==)
          9 LOAD_FAST                0 (0)
         12 STORE_ATTR               0 (0)
         15 LOAD_CONST               0 (0)
         18 RETURN_VALUE
"""

def _f(a):
    print(a)
    return 1

dis_f = u"""\
%3d           0 LOAD_GLOBAL              0 (print)
              3 LOAD_FAST                0 (a)
              6 CALL_FUNCTION            1 (1 positional, 0 keyword pair)
              9 POP_TOP

%3d          10 LOAD_CONST               1 (1)
             13 RETURN_VALUE
""" % (_f.__code__.co_firstlineno + 1,
       _f.__code__.co_firstlineno + 2)


dis_f_co_code = u"""\
          0 LOAD_GLOBAL              0 (0)
          3 LOAD_FAST                0 (0)
          6 CALL_FUNCTION            1 (1 positional, 0 keyword pair)
          9 POP_TOP
         10 LOAD_CONST               1 (1)
         13 RETURN_VALUE
"""


def bug708901():
    for res in range(1,
                     10):
        pass

dis_bug708901 = u"""\
%3d           0 SETUP_LOOP              23 (to 26)
              3 LOAD_GLOBAL              0 (range)
              6 LOAD_CONST               1 (1)

%3d           9 LOAD_CONST               2 (10)
             12 CALL_FUNCTION            2 (2 positional, 0 keyword pair)
             15 GET_ITER
        >>   16 FOR_ITER                 6 (to 25)
             19 STORE_FAST               0 (res)

%3d          22 JUMP_ABSOLUTE           16
        >>   25 POP_BLOCK
        >>   26 LOAD_CONST               0 (None)
             29 RETURN_VALUE
""" % (bug708901.__code__.co_firstlineno + 1,
       bug708901.__code__.co_firstlineno + 2,
       bug708901.__code__.co_firstlineno + 3)


def bug1333982(x=[]):
    assert 0, ([s for s in x] +
              1)
    pass

dis_bug1333982 = u"""\
%3d           0 LOAD_CONST               1 (0)
              3 POP_JUMP_IF_TRUE        41
              6 LOAD_GLOBAL              0 (AssertionError)
              9 BUILD_LIST               0
             12 LOAD_FAST                0 (x)
             15 GET_ITER
        >>   16 FOR_ITER                12 (to 31)
             19 STORE_FAST               1 (s)
             22 LOAD_FAST                1 (s)
             25 LIST_APPEND              2
             28 JUMP_ABSOLUTE           16

%3d     >>   31 LOAD_CONST               2 (1)
             34 BINARY_ADD
             35 CALL_FUNCTION            1 (1 positional, 0 keyword pair)
             38 RAISE_VARARGS            1

%3d     >>   41 LOAD_CONST               0 (None)
             44 RETURN_VALUE
""" % (bug1333982.__code__.co_firstlineno + 1,
       bug1333982.__code__.co_firstlineno + 2,
       bug1333982.__code__.co_firstlineno + 3)

_BIG_LINENO_FORMAT = u"""\
%3d           0 LOAD_GLOBAL              0 (spam)
              3 POP_TOP
              4 LOAD_CONST               0 (None)
              7 RETURN_VALUE
"""

dis_module_contents = """\

# A simple module for testing the dis module.

def f(): pass
def g(): pass
"""

dis_module_expected_results = u"""\
Disassembly of f:
  4           0 LOAD_CONST               0 (None)
              3 RETURN_VALUE

Disassembly of g:
  5           0 LOAD_CONST               0 (None)
              3 RETURN_VALUE

"""

expr_str = u"x + 1"

dis_expr_str = u"""\
  1           0 LOAD_NAME                0 (x)
              3 LOAD_CONST               0 (1)
              6 BINARY_ADD
              7 RETURN_VALUE
"""

simple_stmt_str = u"x = x + 1"

dis_simple_stmt_str = u"""\
  1           0 LOAD_NAME                0 (x)
              3 LOAD_CONST               0 (1)
              6 BINARY_ADD
              7 STORE_NAME               0 (x)
             10 LOAD_CONST               1 (None)
             13 RETURN_VALUE
"""

compound_stmt_str = u"""\
x = 0
while 1:
    x += 1"""
# Trailing newline has been deliberately omitted

dis_compound_stmt_str = u"""\
  1           0 LOAD_CONST               0 (0)
              3 STORE_NAME               0 (x)

  2           6 SETUP_LOOP              14 (to 23)

  3     >>    9 LOAD_NAME                0 (x)
             12 LOAD_CONST               1 (1)
             15 INPLACE_ADD
             16 STORE_NAME               0 (x)
             19 JUMP_ABSOLUTE            9
             22 POP_BLOCK
        >>   23 LOAD_CONST               2 (None)
             26 RETURN_VALUE
"""

dis_traceback = u"""\
%3d           0 SETUP_EXCEPT            12 (to 15)

%3d           3 LOAD_CONST               1 (1)
              6 LOAD_CONST               2 (0)
    -->       9 BINARY_DIVIDE
             10 POP_TOP
             11 POP_BLOCK
             12 JUMP_FORWARD            35 (to 50)

%3d     >>   15 DUP_TOP
             16 LOAD_GLOBAL              0 (Exception)
             19 COMPARE_OP              10 (exception match)
             22 POP_JUMP_IF_FALSE       49
             25 POP_TOP
             26 STORE_FAST               0 (e)
             29 POP_TOP

%3d          30 LOAD_GLOBAL              1 (sys)
             33 LOAD_ATTR                2 (exc_info)
             36 CALL_FUNCTION            0 (0 positional, 0 keyword pair)
             39 LOAD_CONST               3 (2)
             42 BINARY_SUBSCR
             43 STORE_FAST               1 (tb)
             46 JUMP_FORWARD             1 (to 50)
        >>   49 END_FINALLY

%3d     >>   50 LOAD_FAST                1 (tb)
             53 RETURN_VALUE
""" % (TRACEBACK_CODE.co_firstlineno + 1,
       TRACEBACK_CODE.co_firstlineno + 2,
       TRACEBACK_CODE.co_firstlineno + 3,
       TRACEBACK_CODE.co_firstlineno + 4,
       TRACEBACK_CODE.co_firstlineno + 5)

def _g(x):
    yield x

class DisTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.maxDiff = None
        super(DisTests, self).__init__(*args, **kwargs)

    def get_disassembly(self, func, lasti=-1, wrapper=True):
        # We want to test the default printing behaviour, not the file arg
        output = io.StringIO()
        with redirect_stdout(output):
            if wrapper:
                dis3.dis(func)
            else:
                dis3.disassemble(func, lasti)
        return output.getvalue()

    def get_disassemble_as_string(self, func, lasti=-1):
        return self.get_disassembly(func, lasti, False)

    def strip_addresses(self, text):
        return re.sub(r'\b0x[0-9A-Fa-f]+\b', '0x...', text)

    def do_disassembly_test(self, func, expected):
        got = self.get_disassembly(func)
        if got != expected:
            got = self.strip_addresses(got)
        self.assertEqual(got, expected)
    # nose is a tad too zealous
    do_disassembly_test.__test__ = False

    def test_opmap(self):
        self.assertEqual(dis3.opmap["NOP"], 9)
        self.assertIn(dis3.opmap["LOAD_CONST"], dis3.hasconst)
        self.assertIn(dis3.opmap["STORE_NAME"], dis3.hasname)

    def test_opname(self):
        self.assertEqual(dis3.opname[dis3.opmap["LOAD_FAST"]], "LOAD_FAST")

    def test_boundaries(self):
        self.assertEqual(dis3.opmap["EXTENDED_ARG"], dis3.EXTENDED_ARG)
        self.assertEqual(dis3.opmap["STORE_NAME"], dis3.HAVE_ARGUMENT)

    def test_dis(self):
        self.do_disassembly_test(_f, dis_f)

    def test_bug_708901(self):
        self.do_disassembly_test(bug708901, dis_bug708901)

    def test_bug_1333982(self):
        # This one is checking bytecodes generated for an `assert` statement,
        # so fails if the tests are run with -O.  Skip this test then.
        if not __debug__:
            self.skipTest('need asserts, run without -O')

        self.do_disassembly_test(bug1333982, dis_bug1333982)

    def test_big_linenos(self):
        def func(count):
            namespace = {}
            func = "def foo():\n " + "".join(["\n "] * count + ["spam\n"])
            exec(func, namespace)
            return namespace['foo']

        # Test all small ranges
        for i in range(1, 300):
            expected = _BIG_LINENO_FORMAT % (i + 2)
            self.do_disassembly_test(func(i), expected)

        # Test some larger ranges too
        for i in range(300, 5000, 10):
            expected = _BIG_LINENO_FORMAT % (i + 2)
            self.do_disassembly_test(func(i), expected)

        dis_module = types.ModuleType('dis_module')
        exec dis_module_contents in dis_module.__dict__
        self.do_disassembly_test(dis_module, dis_module_expected_results)

    def test_disassemble_str(self):
        self.do_disassembly_test(expr_str, dis_expr_str)
        self.do_disassembly_test(simple_stmt_str, dis_simple_stmt_str)
        self.do_disassembly_test(compound_stmt_str, dis_compound_stmt_str)

    def test_disassemble_bytes(self):
        self.do_disassembly_test(_f.__code__.co_code, dis_f_co_code)

    def test_disassemble_method(self):
        self.do_disassembly_test(_C(1).__init__, dis_c_instance_method)

    def test_disassemble_method_bytes(self):
        method_bytecode = _C(1).__init__.__code__.co_code
        self.do_disassembly_test(method_bytecode, dis_c_instance_method_bytes)

    def test_disassemble_generator(self):
        gen_func_disas = self.get_disassembly(_g)  # Disassemble generator function
        gen_disas = self.get_disassembly(_g(1))  # Disassemble generator itself
        self.assertEqual(gen_disas, gen_func_disas)

    def test_dis_none(self):
        try:
            del sys.last_traceback
        except AttributeError:
            pass
        self.assertRaises(RuntimeError, dis3.dis, None)

    def test_dis_traceback(self):
        try:
            del sys.last_traceback
        except AttributeError:
            pass

        try:
            1/0
        except Exception as e:
            tb = sys.exc_info()[2]
            sys.last_traceback = tb

        tb_dis = self.get_disassemble_as_string(tb.tb_frame.f_code, tb.tb_lasti)
        self.do_disassembly_test(None, tb_dis)

    def test_dis_object(self):
        self.assertRaises(TypeError, dis3.dis, object())

class DisWithFileTests(DisTests):

    # Run the tests again, using the file arg instead of print
    def get_disassembly(self, func, lasti=-1, wrapper=True):
        output = io.StringIO()
        if wrapper:
            dis3.dis(func, file=output)
        else:
            dis3.disassemble(func, lasti, file=output)
        return output.getvalue()



code_info_code_info = """\
Name:              code_info
Filename:          (.*)
Argument count:    1
Number of locals:  1
Stack size:        3
Flags:             OPTIMIZED, NEWLOCALS, NOFREE, FUTURE_PRINT_FUNCTION
Constants:
   0: %r
Names:
   0: _format_code_info
   1: _get_code_object
Variable names:
   0: x""" % (('Formatted details of methods, functions, or code.',)
              if sys.flags.optimize < 2 else (None,))

@staticmethod
def tricky(x, y, z, c, d=[], e=True, *args, **kwds):
    def f(c=c):
        print(x, y, z, c, d, e, f)
    yield x, y, z, c, d, e, f

code_info_tricky = """\
Name:              tricky
Filename:          (.*)
Argument count:    6
Number of locals:  8
Stack size:        7
Flags:             OPTIMIZED, NEWLOCALS, VARARGS, VARKEYWORDS, GENERATOR, FUTURE_PRINT_FUNCTION
Constants:
   0: None
   1: <code object f at (.*), file "(.*)", line (.*)>
Variable names:
   0: x
   1: y
   2: z
   3: c
   4: d
   5: e
   6: args
   7: kwds
Cell variables:
   0: [edfxyz]
   1: [edfxyz]
   2: [edfxyz]
   3: [edfxyz]
   4: [edfxyz]
   5: [edfxyz]"""
# NOTE: the order of the cell variables above depends on dictionary order!

co_tricky_nested_f = tricky.__func__.__code__.co_consts[1]

code_info_tricky_nested_f = """\
Name:              f
Filename:          (.*)
Argument count:    1
Number of locals:  1
Stack size:        8
Flags:             OPTIMIZED, NEWLOCALS, NESTED, FUTURE_PRINT_FUNCTION
Constants:
   0: None
Names:
   0: print
Variable names:
   0: c
Free variables:
   0: [edfxyz]
   1: [edfxyz]
   2: [edfxyz]
   3: [edfxyz]
   4: [edfxyz]
   5: [edfxyz]"""

code_info_expr_str = """\
Name:              <module>
Filename:          <disassembly>
Argument count:    0
Number of locals:  0
Stack size:        2
Flags:             NOFREE, FUTURE_PRINT_FUNCTION
Constants:
   0: 1
Names:
   0: x"""

code_info_simple_stmt_str = """\
Name:              <module>
Filename:          <disassembly>
Argument count:    0
Number of locals:  0
Stack size:        2
Flags:             NOFREE, FUTURE_PRINT_FUNCTION
Constants:
   0: 1
   1: None
Names:
   0: x"""

code_info_compound_stmt_str = """\
Name:              <module>
Filename:          <disassembly>
Argument count:    0
Number of locals:  0
Stack size:        2
Flags:             NOFREE, FUTURE_PRINT_FUNCTION
Constants:
   0: 0
   1: 1
   2: None
Names:
   0: x"""

class CodeInfoTests(unittest.TestCase):
    test_pairs = [
      (dis3.code_info, code_info_code_info),
      (tricky, code_info_tricky),
      (co_tricky_nested_f, code_info_tricky_nested_f),
      (expr_str, code_info_expr_str),
      (simple_stmt_str, code_info_simple_stmt_str),
      (compound_stmt_str, code_info_compound_stmt_str),
    ]

    def __init__(self, *args, **kwargs):
        self.maxDiff = None
        super(CodeInfoTests, self).__init__(*args, **kwargs)

    def test_code_info(self):
        for x, expected in self.test_pairs:
            self.assertRegexpMatches(dis3.code_info(x), expected)

    def test_show_code(self):
        for x, expected in self.test_pairs:
            with captured_stdout() as output:
                dis3.show_code(x)
            self.assertRegexpMatches(output.getvalue(), expected+"\n")
            output = io.StringIO()
            dis3.show_code(x, file=output)
            self.assertRegexpMatches(output.getvalue(), expected)

    def test_code_info_object(self):
        self.assertRaises(TypeError, dis3.code_info, object())

    def test_pretty_flags_no_flags(self):
        self.assertEqual(dis3.pretty_flags(0), '0x0')


# Fodder for instruction introspection tests
#   Editing any of these may require recalculating the expected output
def outer(a=1, b=2):
    def f(c=3, d=4):
        def inner(e=5, f=6):
            print(a, b, c, d, e, f)
        print(a, b, c, d)
        return inner
    print(a, b, u'', 1, [], {}, u"Hello world!")
    return f

def jumpy():
    # This won't actually run (but that's OK, we only disassemble it)
    for i in range(10):
        print(i)
        if i < 4:
            continue
        if i > 6:
            break
    else:
        print(u"I can haz else clause?")
    while i:
        print(i)
        i -= 1
        if i > 6:
            continue
        if i < 4:
            break
    else:
        print(u"Who let lolcatz into this test suite?")
    try:
        1 / 0
    except ZeroDivisionError:
        print(u"Here we go, here we go, here we go...")
    else:
        with i as dodgy:
            print(u"Never reach this")
    finally:
        print(u"OK, now we're done")

# End fodder for opinfo generation tests
expected_outer_line = 1
_line_offset = outer.__code__.co_firstlineno - 1
code_object_f = outer.__code__.co_consts[3]
expected_f_line = code_object_f.co_firstlineno - _line_offset
code_object_inner = code_object_f.co_consts[3]
expected_inner_line = code_object_inner.co_firstlineno - _line_offset
expected_jumpy_line = 1

# The following lines are useful to regenerate the expected results after
# either the fodder is modified or the bytecode generation changes
# After regeneration, update the references to code_object_f and
# code_object_inner before rerunning the tests

# _instructions = dis3.get_instructions(outer, first_line=expected_outer_line)
# print(u'expected_opinfo_outer = [\n  ',
#       u',\n  '.join(map(str, _instructions)), u',\n]', sep=u'')
# _instructions = dis3.get_instructions(outer(), first_line=expected_f_line)
# print(u'expected_opinfo_f = [\n  ',
#       u',\n  '.join(map(str, _instructions)), u',\n]', sep=u'')
# _instructions = dis3.get_instructions(outer()(), first_line=expected_inner_line)
# print(u'expected_opinfo_inner = [\n  ',
#       u',\n  '.join(map(str, _instructions)), u',\n]', sep=u'')
# _instructions = dis3.get_instructions(jumpy, first_line=expected_jumpy_line)
# print(u'expected_opinfo_jumpy = [\n  ',
#       u',\n  '.join(map(str, _instructions)), u',\n]', sep=u'')


Instruction = dis3.Instruction
expected_opinfo_outer = [
  Instruction(opname='LOAD_CONST', opcode=100, arg=1, argval=3, argrepr='3', offset=0, starts_line=2, is_jump_target=False),
  Instruction(opname='LOAD_CONST', opcode=100, arg=2, argval=4, argrepr='4', offset=3, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_CLOSURE', opcode=135, arg=0, argval='a', argrepr='a', offset=6, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_CLOSURE', opcode=135, arg=1, argval='b', argrepr='b', offset=9, starts_line=None, is_jump_target=False),
  Instruction(opname='BUILD_TUPLE', opcode=102, arg=2, argval=2, argrepr='', offset=12, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_CONST', opcode=100, arg=3, argval=code_object_f, argrepr=repr(code_object_f), offset=15, starts_line=None, is_jump_target=False),
  Instruction(opname='MAKE_CLOSURE', opcode=134, arg=2, argval=2, argrepr='', offset=18, starts_line=None, is_jump_target=False),
  Instruction(opname='STORE_FAST', opcode=125, arg=2, argval='f', argrepr='f', offset=21, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_GLOBAL', opcode=116, arg=0, argval='print', argrepr='print', offset=24, starts_line=7, is_jump_target=False),
  Instruction(opname='LOAD_DEREF', opcode=136, arg=0, argval='a', argrepr='a', offset=27, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_DEREF', opcode=136, arg=1, argval='b', argrepr='b', offset=30, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_CONST', opcode=100, arg=4, argval=u'', argrepr="u''", offset=33, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_CONST', opcode=100, arg=5, argval=1, argrepr='1', offset=36, starts_line=None, is_jump_target=False),
  Instruction(opname='BUILD_LIST', opcode=103, arg=0, argval=0, argrepr='', offset=39, starts_line=None, is_jump_target=False),
  Instruction(opname='BUILD_MAP', opcode=105, arg=0, argval=0, argrepr='', offset=42, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_CONST', opcode=100, arg=6, argval=u'Hello world!', argrepr="u'Hello world!'", offset=45, starts_line=None, is_jump_target=False),
  Instruction(opname='CALL_FUNCTION', opcode=131, arg=7, argval=7, argrepr='7 positional, 0 keyword pair', offset=48, starts_line=None, is_jump_target=False),
  Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=51, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_FAST', opcode=124, arg=2, argval='f', argrepr='f', offset=52, starts_line=8, is_jump_target=False),
  Instruction(opname='RETURN_VALUE', opcode=83, arg=None, argval=None, argrepr='', offset=55, starts_line=None, is_jump_target=False),
]

expected_opinfo_f = [
  Instruction(opname='LOAD_CONST', opcode=100, arg=1, argval=5, argrepr='5', offset=0, starts_line=3, is_jump_target=False),
  Instruction(opname='LOAD_CONST', opcode=100, arg=2, argval=6, argrepr='6', offset=3, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_CLOSURE', opcode=135, arg=2, argval='a', argrepr='a', offset=6, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_CLOSURE', opcode=135, arg=3, argval='b', argrepr='b', offset=9, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_CLOSURE', opcode=135, arg=0, argval='c', argrepr='c', offset=12, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_CLOSURE', opcode=135, arg=1, argval='d', argrepr='d', offset=15, starts_line=None, is_jump_target=False),
  Instruction(opname='BUILD_TUPLE', opcode=102, arg=4, argval=4, argrepr='', offset=18, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_CONST', opcode=100, arg=3, argval=code_object_inner, argrepr=repr(code_object_inner), offset=21, starts_line=None, is_jump_target=False),
  Instruction(opname='MAKE_CLOSURE', opcode=134, arg=2, argval=2, argrepr='', offset=24, starts_line=None, is_jump_target=False),
  Instruction(opname='STORE_FAST', opcode=125, arg=2, argval='inner', argrepr='inner', offset=27, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_GLOBAL', opcode=116, arg=0, argval='print', argrepr='print', offset=30, starts_line=5, is_jump_target=False),
  Instruction(opname='LOAD_DEREF', opcode=136, arg=2, argval='a', argrepr='a', offset=33, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_DEREF', opcode=136, arg=3, argval='b', argrepr='b', offset=36, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_DEREF', opcode=136, arg=0, argval='c', argrepr='c', offset=39, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_DEREF', opcode=136, arg=1, argval='d', argrepr='d', offset=42, starts_line=None, is_jump_target=False),
  Instruction(opname='CALL_FUNCTION', opcode=131, arg=4, argval=4, argrepr='4 positional, 0 keyword pair', offset=45, starts_line=None, is_jump_target=False),
  Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=48, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_FAST', opcode=124, arg=2, argval='inner', argrepr='inner', offset=49, starts_line=6, is_jump_target=False),
  Instruction(opname='RETURN_VALUE', opcode=83, arg=None, argval=None, argrepr='', offset=52, starts_line=None, is_jump_target=False),
]

expected_opinfo_inner = [
  Instruction(opname='LOAD_GLOBAL', opcode=116, arg=0, argval='print', argrepr='print', offset=0, starts_line=4, is_jump_target=False),
  Instruction(opname='LOAD_DEREF', opcode=136, arg=0, argval='a', argrepr='a', offset=3, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_DEREF', opcode=136, arg=1, argval='b', argrepr='b', offset=6, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_DEREF', opcode=136, arg=2, argval='c', argrepr='c', offset=9, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_DEREF', opcode=136, arg=3, argval='d', argrepr='d', offset=12, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_FAST', opcode=124, arg=0, argval='e', argrepr='e', offset=15, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_FAST', opcode=124, arg=1, argval='f', argrepr='f', offset=18, starts_line=None, is_jump_target=False),
  Instruction(opname='CALL_FUNCTION', opcode=131, arg=6, argval=6, argrepr='6 positional, 0 keyword pair', offset=21, starts_line=None, is_jump_target=False),
  Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=24, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_CONST', opcode=100, arg=0, argval=None, argrepr='None', offset=25, starts_line=None, is_jump_target=False),
  Instruction(opname='RETURN_VALUE', opcode=83, arg=None, argval=None, argrepr='', offset=28, starts_line=None, is_jump_target=False),
]

expected_opinfo_jumpy = [
  Instruction(opname='SETUP_LOOP', opcode=120, arg=74, argval=77, argrepr='to 77', offset=0, starts_line=3, is_jump_target=False),
  Instruction(opname='LOAD_GLOBAL', opcode=116, arg=0, argval='range', argrepr='range', offset=3, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_CONST', opcode=100, arg=1, argval=10, argrepr='10', offset=6, starts_line=None, is_jump_target=False),
  Instruction(opname='CALL_FUNCTION', opcode=131, arg=1, argval=1, argrepr='1 positional, 0 keyword pair', offset=9, starts_line=None, is_jump_target=False),
  Instruction(opname='GET_ITER', opcode=68, arg=None, argval=None, argrepr='', offset=12, starts_line=None, is_jump_target=False),
  Instruction(opname='FOR_ITER', opcode=93, arg=50, argval=66, argrepr='to 66', offset=13, starts_line=None, is_jump_target=True),
  Instruction(opname='STORE_FAST', opcode=125, arg=0, argval='i', argrepr='i', offset=16, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_GLOBAL', opcode=116, arg=1, argval='print', argrepr='print', offset=19, starts_line=4, is_jump_target=False),
  Instruction(opname='LOAD_FAST', opcode=124, arg=0, argval='i', argrepr='i', offset=22, starts_line=None, is_jump_target=False),
  Instruction(opname='CALL_FUNCTION', opcode=131, arg=1, argval=1, argrepr='1 positional, 0 keyword pair', offset=25, starts_line=None, is_jump_target=False),
  Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=28, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_FAST', opcode=124, arg=0, argval='i', argrepr='i', offset=29, starts_line=5, is_jump_target=False),
  Instruction(opname='LOAD_CONST', opcode=100, arg=2, argval=4, argrepr='4', offset=32, starts_line=None, is_jump_target=False),
  Instruction(opname='COMPARE_OP', opcode=107, arg=0, argval='<', argrepr='<', offset=35, starts_line=None, is_jump_target=False),
  Instruction(opname='POP_JUMP_IF_FALSE', opcode=114, arg=47, argval=47, argrepr='', offset=38, starts_line=None, is_jump_target=False),
  Instruction(opname='JUMP_ABSOLUTE', opcode=113, arg=13, argval=13, argrepr='', offset=41, starts_line=6, is_jump_target=False),
  Instruction(opname='JUMP_FORWARD', opcode=110, arg=0, argval=47, argrepr='to 47', offset=44, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_FAST', opcode=124, arg=0, argval='i', argrepr='i', offset=47, starts_line=7, is_jump_target=True),
  Instruction(opname='LOAD_CONST', opcode=100, arg=3, argval=6, argrepr='6', offset=50, starts_line=None, is_jump_target=False),
  Instruction(opname='COMPARE_OP', opcode=107, arg=4, argval='>', argrepr='>', offset=53, starts_line=None, is_jump_target=False),
  Instruction(opname='POP_JUMP_IF_FALSE', opcode=114, arg=13, argval=13, argrepr='', offset=56, starts_line=None, is_jump_target=False),
  Instruction(opname='BREAK_LOOP', opcode=80, arg=None, argval=None, argrepr='', offset=59, starts_line=8, is_jump_target=False),
  Instruction(opname='JUMP_ABSOLUTE', opcode=113, arg=13, argval=13, argrepr='', offset=60, starts_line=None, is_jump_target=False),
  Instruction(opname='JUMP_ABSOLUTE', opcode=113, arg=13, argval=13, argrepr='', offset=63, starts_line=None, is_jump_target=False),
  Instruction(opname='POP_BLOCK', opcode=87, arg=None, argval=None, argrepr='', offset=66, starts_line=None, is_jump_target=True),
  Instruction(opname='LOAD_GLOBAL', opcode=116, arg=1, argval='print', argrepr='print', offset=67, starts_line=10, is_jump_target=False),
  Instruction(opname='LOAD_CONST', opcode=100, arg=4, argval=u'I can haz else clause?', argrepr="u'I can haz else clause?'", offset=70, starts_line=None, is_jump_target=False),
  Instruction(opname='CALL_FUNCTION', opcode=131, arg=1, argval=1, argrepr='1 positional, 0 keyword pair', offset=73, starts_line=None, is_jump_target=False),
  Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=76, starts_line=None, is_jump_target=False),
  Instruction(opname='SETUP_LOOP', opcode=120, arg=74, argval=154, argrepr='to 154', offset=77, starts_line=11, is_jump_target=True),
  Instruction(opname='LOAD_FAST', opcode=124, arg=0, argval='i', argrepr='i', offset=80, starts_line=None, is_jump_target=True),
  Instruction(opname='POP_JUMP_IF_FALSE', opcode=114, arg=143, argval=143, argrepr='', offset=83, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_GLOBAL', opcode=116, arg=1, argval='print', argrepr='print', offset=86, starts_line=12, is_jump_target=False),
  Instruction(opname='LOAD_FAST', opcode=124, arg=0, argval='i', argrepr='i', offset=89, starts_line=None, is_jump_target=False),
  Instruction(opname='CALL_FUNCTION', opcode=131, arg=1, argval=1, argrepr='1 positional, 0 keyword pair', offset=92, starts_line=None, is_jump_target=False),
  Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=95, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_FAST', opcode=124, arg=0, argval='i', argrepr='i', offset=96, starts_line=13, is_jump_target=False),
  Instruction(opname='LOAD_CONST', opcode=100, arg=5, argval=1, argrepr='1', offset=99, starts_line=None, is_jump_target=False),
  Instruction(opname='INPLACE_SUBTRACT', opcode=56, arg=None, argval=None, argrepr='', offset=102, starts_line=None, is_jump_target=False),
  Instruction(opname='STORE_FAST', opcode=125, arg=0, argval='i', argrepr='i', offset=103, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_FAST', opcode=124, arg=0, argval='i', argrepr='i', offset=106, starts_line=14, is_jump_target=False),
  Instruction(opname='LOAD_CONST', opcode=100, arg=3, argval=6, argrepr='6', offset=109, starts_line=None, is_jump_target=False),
  Instruction(opname='COMPARE_OP', opcode=107, arg=4, argval='>', argrepr='>', offset=112, starts_line=None, is_jump_target=False),
  Instruction(opname='POP_JUMP_IF_FALSE', opcode=114, arg=124, argval=124, argrepr='', offset=115, starts_line=None, is_jump_target=False),
  Instruction(opname='JUMP_ABSOLUTE', opcode=113, arg=80, argval=80, argrepr='', offset=118, starts_line=15, is_jump_target=False),
  Instruction(opname='JUMP_FORWARD', opcode=110, arg=0, argval=124, argrepr='to 124', offset=121, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_FAST', opcode=124, arg=0, argval='i', argrepr='i', offset=124, starts_line=16, is_jump_target=True),
  Instruction(opname='LOAD_CONST', opcode=100, arg=2, argval=4, argrepr='4', offset=127, starts_line=None, is_jump_target=False),
  Instruction(opname='COMPARE_OP', opcode=107, arg=0, argval='<', argrepr='<', offset=130, starts_line=None, is_jump_target=False),
  Instruction(opname='POP_JUMP_IF_FALSE', opcode=114, arg=80, argval=80, argrepr='', offset=133, starts_line=None, is_jump_target=False),
  Instruction(opname='BREAK_LOOP', opcode=80, arg=None, argval=None, argrepr='', offset=136, starts_line=17, is_jump_target=False),
  Instruction(opname='JUMP_ABSOLUTE', opcode=113, arg=80, argval=80, argrepr='', offset=137, starts_line=None, is_jump_target=False),
  Instruction(opname='JUMP_ABSOLUTE', opcode=113, arg=80, argval=80, argrepr='', offset=140, starts_line=None, is_jump_target=False),
  Instruction(opname='POP_BLOCK', opcode=87, arg=None, argval=None, argrepr='', offset=143, starts_line=None, is_jump_target=True),
  Instruction(opname='LOAD_GLOBAL', opcode=116, arg=1, argval='print', argrepr='print', offset=144, starts_line=19, is_jump_target=False),
  Instruction(opname='LOAD_CONST', opcode=100, arg=6, argval=u'Who let lolcatz into this test suite?', argrepr="u'Who let lolcatz into this test suite?'", offset=147, starts_line=None, is_jump_target=False),
  Instruction(opname='CALL_FUNCTION', opcode=131, arg=1, argval=1, argrepr='1 positional, 0 keyword pair', offset=150, starts_line=None, is_jump_target=False),
  Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=153, starts_line=None, is_jump_target=False),
  Instruction(opname='SETUP_FINALLY', opcode=122, arg=71, argval=228, argrepr='to 228', offset=154, starts_line=20, is_jump_target=True),
  Instruction(opname='SETUP_EXCEPT', opcode=121, arg=12, argval=172, argrepr='to 172', offset=157, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_CONST', opcode=100, arg=5, argval=1, argrepr='1', offset=160, starts_line=21, is_jump_target=False),
  Instruction(opname='LOAD_CONST', opcode=100, arg=7, argval=0, argrepr='0', offset=163, starts_line=None, is_jump_target=False),
  Instruction(opname='BINARY_DIVIDE', opcode=21, arg=None, argval=None, argrepr='', offset=166, starts_line=None, is_jump_target=False),
  Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=167, starts_line=None, is_jump_target=False),
  Instruction(opname='POP_BLOCK', opcode=87, arg=None, argval=None, argrepr='', offset=168, starts_line=None, is_jump_target=False),
  Instruction(opname='JUMP_FORWARD', opcode=110, arg=27, argval=199, argrepr='to 199', offset=169, starts_line=None, is_jump_target=False),
  Instruction(opname='DUP_TOP', opcode=4, arg=None, argval=None, argrepr='', offset=172, starts_line=22, is_jump_target=True),
  Instruction(opname='LOAD_GLOBAL', opcode=116, arg=2, argval='ZeroDivisionError', argrepr='ZeroDivisionError', offset=173, starts_line=None, is_jump_target=False),
  Instruction(opname='COMPARE_OP', opcode=107, arg=10, argval='exception match', argrepr='exception match', offset=176, starts_line=None, is_jump_target=False),
  Instruction(opname='POP_JUMP_IF_FALSE', opcode=114, arg=198, argval=198, argrepr='', offset=179, starts_line=None, is_jump_target=False),
  Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=182, starts_line=None, is_jump_target=False),
  Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=183, starts_line=None, is_jump_target=False),
  Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=184, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_GLOBAL', opcode=116, arg=1, argval='print', argrepr='print', offset=185, starts_line=23, is_jump_target=False),
  Instruction(opname='LOAD_CONST', opcode=100, arg=8, argval=u'Here we go, here we go, here we go...', argrepr="u'Here we go, here we go, here we go...'", offset=188, starts_line=None, is_jump_target=False),
  Instruction(opname='CALL_FUNCTION', opcode=131, arg=1, argval=1, argrepr='1 positional, 0 keyword pair', offset=191, starts_line=None, is_jump_target=False),
  Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=194, starts_line=None, is_jump_target=False),
  Instruction(opname='JUMP_FORWARD', opcode=110, arg=26, argval=224, argrepr='to 224', offset=195, starts_line=None, is_jump_target=False),
  Instruction(opname='END_FINALLY', opcode=88, arg=None, argval=None, argrepr='', offset=198, starts_line=None, is_jump_target=True),
  Instruction(opname='LOAD_FAST', opcode=124, arg=0, argval='i', argrepr='i', offset=199, starts_line=25, is_jump_target=True),
  Instruction(opname='SETUP_WITH', opcode=143, arg=17, argval=222, argrepr='to 222', offset=202, starts_line=None, is_jump_target=False),
  Instruction(opname='STORE_FAST', opcode=125, arg=1, argval='dodgy', argrepr='dodgy', offset=205, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_GLOBAL', opcode=116, arg=1, argval='print', argrepr='print', offset=208, starts_line=26, is_jump_target=False),
  Instruction(opname='LOAD_CONST', opcode=100, arg=9, argval=u'Never reach this', argrepr="u'Never reach this'", offset=211, starts_line=None, is_jump_target=False),
  Instruction(opname='CALL_FUNCTION', opcode=131, arg=1, argval=1, argrepr='1 positional, 0 keyword pair', offset=214, starts_line=None, is_jump_target=False),
  Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=217, starts_line=None, is_jump_target=False),
  Instruction(opname='POP_BLOCK', opcode=87, arg=None, argval=None, argrepr='', offset=218, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_CONST', opcode=100, arg=0, argval=None, argrepr='None', offset=219, starts_line=None, is_jump_target=False),
  Instruction(opname='WITH_CLEANUP', opcode=81, arg=None, argval=None, argrepr='', offset=222, starts_line=None, is_jump_target=True),
  Instruction(opname='END_FINALLY', opcode=88, arg=None, argval=None, argrepr='', offset=223, starts_line=None, is_jump_target=False),
  Instruction(opname='POP_BLOCK', opcode=87, arg=None, argval=None, argrepr='', offset=224, starts_line=None, is_jump_target=True),
  Instruction(opname='LOAD_CONST', opcode=100, arg=0, argval=None, argrepr='None', offset=225, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_GLOBAL', opcode=116, arg=1, argval='print', argrepr='print', offset=228, starts_line=28, is_jump_target=True),
  Instruction(opname='LOAD_CONST', opcode=100, arg=10, argval=u"OK, now we're done", argrepr='u"OK, now we\'re done"', offset=231, starts_line=None, is_jump_target=False),
  Instruction(opname='CALL_FUNCTION', opcode=131, arg=1, argval=1, argrepr='1 positional, 0 keyword pair', offset=234, starts_line=None, is_jump_target=False),
  Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=237, starts_line=None, is_jump_target=False),
  Instruction(opname='END_FINALLY', opcode=88, arg=None, argval=None, argrepr='', offset=238, starts_line=None, is_jump_target=False),
  Instruction(opname='LOAD_CONST', opcode=100, arg=0, argval=None, argrepr='None', offset=239, starts_line=None, is_jump_target=False),
  Instruction(opname='RETURN_VALUE', opcode=83, arg=None, argval=None, argrepr='', offset=242, starts_line=None, is_jump_target=False),
]

# One last piece of inspect fodder to check the default line number handling
def simple(): pass
expected_opinfo_simple = [
  Instruction(opname='LOAD_CONST', opcode=100, arg=0, argval=None, argrepr='None', offset=0, starts_line=simple.__code__.co_firstlineno, is_jump_target=False),
  Instruction(opname='RETURN_VALUE', opcode=83, arg=None, argval=None, argrepr='', offset=3, starts_line=None, is_jump_target=False)
]


class InstructionTests(BytecodeTestCase):
    def __init__(self, *args, **kwargs):
        self.maxDiff = None
        super(InstructionTests, self).__init__(*args, **kwargs)

    def test_default_first_line(self):
        actual = dis3.get_instructions(simple)
        self.assertEqual(list(actual), expected_opinfo_simple)

    def test_first_line_set_to_None(self):
        actual = dis3.get_instructions(simple, first_line=None)
        self.assertEqual(list(actual), expected_opinfo_simple)

    def test_outer(self):
        actual = dis3.get_instructions(outer, first_line=expected_outer_line)
        self.assertEqual(list(actual), expected_opinfo_outer)

    def test_nested(self):
        with captured_stdout():
            f = outer()
        actual = dis3.get_instructions(f, first_line=expected_f_line)
        self.assertEqual(list(actual), expected_opinfo_f)

    def test_doubly_nested(self):
        with captured_stdout():
            inner = outer()()
        actual = dis3.get_instructions(inner, first_line=expected_inner_line)
        self.assertEqual(list(actual), expected_opinfo_inner)

    def test_jumpy(self):
        actual = dis3.get_instructions(jumpy, first_line=expected_jumpy_line)
        self.assertEqual(list(actual), expected_opinfo_jumpy)

# get_instructions has its own tests above, so can rely on it to validate
# the object oriented API
class BytecodeTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.maxDiff = None
        super(BytecodeTests, self).__init__(*args, **kwargs)

    def test_instantiation_function(self):
        b = dis3.Bytecode(_f)
        self.assertIsInstance(b.codeobj, types.CodeType)

    def test_instantiation_method(self):
        b = dis3.Bytecode(_C(1).__init__)
        self.assertIsInstance(b.codeobj, types.CodeType)

    def test_instantiation_code_string(self):
        b = dis3.Bytecode(u"a=1")
        self.assertIsInstance(b.codeobj, types.CodeType)

    def test_instantiation_code_object(self):
        b = dis3.Bytecode(_f.__code__)
        self.assertIsInstance(b.codeobj, types.CodeType)

    def test_instantiation_object(self):
        self.assertRaises(TypeError, dis3.Bytecode, object())

    def test_iteration_function(self):
        obj = _f
        via_object = list(dis3.Bytecode(obj))
        via_generator = list(dis3.get_instructions(obj))
        self.assertEqual(via_object, via_generator)

    def test_iteration_method(self):
        obj = _C(1).__init__
        via_object = list(dis3.Bytecode(obj))
        via_generator = list(dis3.get_instructions(obj))
        self.assertEqual(via_object, via_generator)

    def test_iteration_code_string(self):
        obj = u"a=1"
        via_object = list(dis3.Bytecode(obj))
        via_generator = list(dis3.get_instructions(obj))
        self.assertEqual(via_object, via_generator)

    def test_iteration_code_object(self):
        obj = _f.__code__
        via_object = list(dis3.Bytecode(obj))
        via_generator = list(dis3.get_instructions(obj))
        self.assertEqual(via_object, via_generator)

    def test_explicit_first_line(self):
        actual = dis3.Bytecode(outer, first_line=expected_outer_line)
        self.assertEqual(list(actual), expected_opinfo_outer)

    def test_source_line_in_disassembly(self):
        # Use the line in the source code
        actual = dis3.Bytecode(simple).dis()[:3]
        expected = u"{:>3}".format(simple.__code__.co_firstlineno)
        self.assertEqual(actual, expected)
        # Use an explicit first line number
        actual = dis3.Bytecode(simple, first_line=350).dis()[:3]
        self.assertEqual(actual, "350")

    def test_info(self):
        for x, expected in CodeInfoTests.test_pairs:
            b = dis3.Bytecode(x)
            self.assertRegexpMatches(b.info(), expected)

    def test_disassembled(self):
        actual = dis3.Bytecode(_f).dis()
        self.assertEqual(actual, dis_f)

    def test_from_traceback(self):
        tb = get_tb()
        b = dis3.Bytecode.from_traceback(tb)
        while tb.tb_next: tb = tb.tb_next

        self.assertEqual(b.current_offset, tb.tb_lasti)

    def test_from_traceback_dis(self):
        tb = get_tb()
        b = dis3.Bytecode.from_traceback(tb)
        self.assertEqual(b.dis(), dis_traceback)

if __name__ == "__main__":
    unittest.main()
