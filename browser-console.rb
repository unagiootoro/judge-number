require "js"
require "stringio"

class BrowserConsoleImpl < StringIO
  attr_accessor :dom_element

  def initialize(dom_element: nil, max_lines: 256)
    super()
    if dom_element
      @dom_element = dom_element
    else
      document = JS.global[:document]
      @dom_element = document.createElement("p")
      document[:body].appendChild(@dom_element)
    end
    @max_lines = max_lines
  end

  def update
    @dom_element[:innerText] = string
  end

  def max_lines
    @max_lines
  end

  def max_lines=(value)
    @max_lines = value
    update
  end

  def print(*obj)
    super(*obj)
    update
  end

  def puts(*obj)
    super(*obj)
    update
  end
end

BrowserConsole = BrowserConsoleImpl.new
