from lab2.search_system import SearchSystem


def main():
    ss = SearchSystem()
    ss.build_from_mml('../search_system_data')

    for w in ['game', 'play', 'character', 'role', 'class', 'interface', 'processor', '']:
        relevance = ss.search(w)
        print(f'"{w}" search results:')
        print(*relevance, sep='\n')
        print()


if __name__ == '__main__':
    main()
