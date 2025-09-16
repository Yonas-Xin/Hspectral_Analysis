ACADEMIC_COLOR = ["#ffffff", "#ff0000", "#ffd000", "#00c8ff", "#1aff00", "#bb00ff"] # 白色、红色、黄色、蓝色、绿色、紫色
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
rgb_colors = [hex_to_rgb(color) for color in ACADEMIC_COLOR]