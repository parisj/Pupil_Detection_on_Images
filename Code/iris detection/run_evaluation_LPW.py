import ellipse_methods as em
import time 
def run():
    with open('LPW.csv') as f:
        lines = f.read()
    
    print(lines)
    lines = lines.split('\n')
    for current, line in enumerate(lines):
        start_time = time.time()
        print(f'reading line: {line}, at index: {current+1} of {len(lines)}')
        em.main_detection(line)
        end_time = time.time()
        time_taken = end_time - start_time
        print (f'Time taken for this video: {time_taken} seconds')
    return True

if __name__ == '__main__':
    run()